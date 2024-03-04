import sys
from tqdm import tqdm
import os 
code_path = 'INSERT_CODE_PATH_HERE' # e.g. '/home/username/ProS' 
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
from clip.model import CLIP
import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.DomainNet import domainnet
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.dataloaders import CuMixloader, BaselineDataset
from data.sampler import BalancedSampler
from utils import utils, GPUmanager
from utils.logger import AverageMeter
from utils.metrics import compute_retrieval_metrics
from PIL import Image
from functools import reduce
from operator import mul

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.clip_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) # 400, 77, 512
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

      
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames) # 400
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        ctx_dim = clip_model.ln_final.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution 
        cfg_imsize = cfg.image_size 
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device) # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) #n n_ctx X，eg. X X X X X

        print(f'Initial context: "{prompt_prefix}"') # 'X X X X X X X X X X X X X X X X'
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # xxxxxxxxx classname .

         # Convert each word in the prompts to numerical values based on the dictionary,
        # with a fixed length of 77 tokens. If the prompt is shorter, it is padded with 0.
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device) # [400,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 400, 77, 512

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 400,16,512

        prefix = self.token_prefix # 400,1,512
        suffix = self.token_suffix # 400,60,512

        
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts # 400,77,512

class PromptedVisionTransformer(nn.Module):
    def __init__(self, config,model:CLIP):
        super(PromptedVisionTransformer, self).__init__()
        self.input_resolution = model.visual.input_resolution
        self.output_dim = model.visual.output_dim
        self.conv1 = model.visual.conv1
        width = self.conv1.out_channels
        self.class_embedding = model.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.ln_post = model.visual.ln_post
        self.proj = model.visual.proj
        self.config = config
        patch_size = self.conv1.kernel_size
        num_tokens = self.config.vp_NUM_TOKENS  # "10"
        self.num_tokens = num_tokens  # number of prompted tokens
        if self.config.vp_PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.config.vp_PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        if self.config.vp_INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim)) # layer, num_token, prompt_dim
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.config.vp_DEEP:  # noqa

                total_d_layer = self.transformer.layers - 1 
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        # after CLS token, all before image patches
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype) # [65, 50, 678] + [50 ,768]

        # x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        
        x = torch.cat((
            x[:, :1, :], # CLS token
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1),
            x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = self.transformer.layers
        # print("yes")
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output) 
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

                hidden_states= self.transformer.resblocks[i](hidden_states)
        return hidden_states

    def forward(self, x, vis=False):
                # this is the default version:
        x = self.incorporate_prompt(x)

        if self.config.vp_DEEP:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.forward_deep_prompt(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        else:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        return x
       
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP):
        super().__init__()
        self.cfg = cfg
        if cfg.training_strategy == 'TP':
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model).to(device)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
            self.text_encoder = TextEncoder(clip_model).to(device)
        else :
            self.text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)
            # self.text_features = clip_model.encode_text(self.text_template)
            self.text_encoder = clip_model.encode_text
        if cfg.training_strategy == 'VP':
            self.image_encoder = PromptedVisionTransformer(cfg, clip_model)
        else :
            self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype 

    def forward(self, image): # bt 3, 244, 244
        image_features = self.image_encoder(image.type(self.dtype)) # batch, 512
        if self.cfg.training_strategy == 'TP':
            prompts = self.prompt_learner() # 400,77,512
            tokenized_prompts = self.tokenized_prompts # 400,77 400类，
            text_features = self.text_encoder(prompts, tokenized_prompts) # class, 512
        else :
            text_features = self.text_encoder(self.text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class Trainer:

    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        if args.dataset == 'Sketchy':
            data_input = sketchy_extended.create_trvalte_splits(args)
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)
        if args.dataset == 'TUBerlin':
            data_input = tuberlin_extended.create_trvalte_splits(args)

        self.tr_classes = data_input['tr_classes']
        self.va_classes = data_input['va_classes']
        self.te_classes = data_input['te_classes']
        semantic_vec = data_input['semantic_vec']
        self.data_splits = data_input['splits']
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        # Image transformations
        self.image_transforms = {
            'train':
                transforms.Compose([
                    transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(im_mean, im_std)
                ]),
            'eval': transforms.Compose([
                transforms.Resize(args.image_size, interpolation=BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                #lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes)  # 生成 类:index 的一个字典
        self.te_dict_class = utils.create_dict_texts(self.tr_classes+self.va_classes+self.te_classes)

        fls_tr = self.data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)
        print(self.dict_doms)
        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)

        data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=self.image_transforms['train'])
        train_sampler = BalancedSampler(domain_ids, args.batch_size // len(tr_domains_unique),
                                        domains_per_batch=len(tr_domains_unique))  # 每个batch的采样都来自同一个domain
        self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)

        data_va_query = BaselineDataset(self.data_splits['query_va'], transforms=self.image_transforms['eval'])
        data_va_gallery = BaselineDataset(self.data_splits['gallery_va'], transforms=self.image_transforms['eval'])
        # data_va_query = BaselineDataset(data_splits['query_va'],)
        # data_va_gallery = BaselineDataset(data_splits['gallery_va'])

        # PyTorch valid loader for query
        self.va_loader_query = DataLoader(dataset=data_va_query, batch_size=args.batch_size , shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
        # PyTorch valid loader for gallery
        self.va_loader_gallery = DataLoader(dataset=data_va_gallery, batch_size=args.batch_size , shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

        print(
            f'#Tr samples:{len(data_train)}; #Val queries:{len(data_va_query)}; #Val gallery samples:{len(data_va_gallery)}.\n')
        print('Loading Done\n')

        self.text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.tr_classes]).to(device)

        self.RG = np.random.default_rng()

        self.build_model()


        if args.dataset=='DomainNet':
            save_folder_name = 'seen-'+args.seen_domain+'_unseen-'+args.holdout_domain+'_x_'+args.gallery_domain
            if not args.include_auxillary_domains:
                save_folder_name += '_noaux'
        elif args.dataset=='Sketchy':
            if args.is_eccv_split:
                save_folder_name = 'eccv_split'
            else:
                save_folder_name = 'random_split'
        else:
            save_folder_name = ''

        if args.dataset=='DomainNet' or (args.dataset=='Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'


        self.suffix =  'e-'+str(args.epochs)+'_es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
                        '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)+'_l2-'+str(args.l2_reg)+\
                        '_seed-'+str(args.seed)

        # exit(0)
        self.path_cp = os.path.join(args.code_path, 'src/algos/PromptTuning/saved_models',args.training_strategy, args.dataset, save_folder_name)
        
        print('Done\n')

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_chkpt_name='init'

        self.resume_from_checkpoint(args.resume_dict)
        

    def do_epoch(self):

        self.model.train()

        batch_time = AverageMeter()
        total_loss = AverageMeter()

        # Start counting time
        time_start = time.time()

        for i, (im, cl, domain_ids) in enumerate(self.train_loader):

            # Transfer im to cuda
            im = im.float().to(device)
            # Get numeric classes
            cls_numeric = torch.from_numpy(utils.numeric_classes(cl, self.dict_clss)).long().to(device)
            # print(cls_numeric.shape)
            self.optimizer.zero_grad()

            logits_per_image = self.model(im)
            loss = F.cross_entropy(logits_per_image, cls_numeric)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss.update(loss.item(), im.size(0))
            
            # time
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end
            
            if (i + 1) % self.args.log_interval == 0:
                print('[Train] Epoch: [{0}/{1}][{2}/{3}]\t'
                        # 'lr:{3:.6f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'net {net.val:.4f} ({net.avg:.4f})\t'
                        .format(self.current_epoch+1, self.args.epochs, i+1, len(self.train_loader), batch_time=batch_time, net=total_loss))
                if self.args.debug_mode == 1:
                    break
            # if (i+1)==5:
            #     break

        return {'net':total_loss.avg}

    def do_training(self):

        print('***Train***')
        for self.current_epoch in range(self.start_epoch, self.args.epochs):

            start = time.time()

            self.adjust_learning_rate()

            loss = self.do_epoch()

            # evaluate on validation set, map_ since map is already there
            print('\n***Validation***')
            if self.args.dataset=='DomainNet':
                te_data = []
                for domain in [self.args.seen_domain, self.args.holdout_domain]:
                    for includeSeenClassinTestGallery in [0, 1]:
                        test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(includeSeenClassinTestGallery)
                        print(test_head_str)

                        splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes, self.va_classes, self.te_classes)
                        splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain, includeSeenClassinTestGallery, self.tr_classes, self.va_classes, self.te_classes)

                        data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=self.image_transforms['eval'])
                        data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=self.image_transforms['eval'])

                        # PyTorch test loader for query
                        te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size*5, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)
                        # PyTorch test loader for gallery
                        te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size*5, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)

                        # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                        result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class)
                        te_data.append(result)
                        
                        out ="mAP@200 = %.4f, Prec@200 = %.4f\n"%(result['mAP@200'], result['prec@200'])
                        print(out)
                map_ = te_data[3]['mAP@200']
                prec = te_data[3]['prec@200']
            else :
                data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

                te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

                print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

                te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class)
                print(te_data)
                out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(te_data[self.map_metric], te_data[self.prec_metric])
           #  valid_data = evaluate(self.va_loader_query, self.va_loader_gallery, self.model, self.te_dict_class)

                map_ = te_data[self.map_metric]
                prec = te_data[self.prec_metric]
            # prec = valid_data[self.prec_metric]
                out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(map_, prec)
           #  valid_data = evaluate(self.va_loader_query, self.va_loader_gallery, self.model, self.te_dict_class)
            print(out)
            end = time.time()
            elapsed = end-start

            print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.7f} mAP:{map_:.4f} prec:{prec:.4f}\n")
           #  exit()
            if map_ > self.best_map:

                self.best_map = map_
                self.early_stop_counter = 0

                model_save_name = 'val_map-'+'{0:.4f}'.format(map_)+'_prec-'+'{0:.4f}'.format(prec)+'_ep-'+str(self.current_epoch+1)+self.suffix
                utils.save_checkpoint({
                                        'epoch':self.current_epoch+1,
                                        'model_state_dict':self.model.state_dict(),
                                        'optimizer_state_dict':self.optimizer.state_dict(),
                                        'best_map':self.best_map,
                                        'corr_prec':prec
                                        }, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
                self.last_chkpt_name = model_save_name

            else:
                self.early_stop_counter += 1
                if self.args.early_stop==self.early_stop_counter:
                    print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
                            f"Early stopping by {self.args.epochs-self.current_epoch-1} epochs.")
                    break

                print(f"Val mAP hasn't improved from {self.best_map:.4f} for {self.early_stop_counter} epoch(s)!\n")

        print('\n***Training and Validation complete***')

    
    def adjust_learning_rate(self, min_lr=1e-6):
        # lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
        # epoch_curr = min(epoch, 20)
        # lr = args.lr * math.pow(0.001, float(epoch_curr)/ 20 )
        lr = self.args.lr * math.pow(1e-3, float(self.current_epoch)/20)
        lr = max(lr, min_lr)
        # print('epoch: {}, lr: {}'.format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def resume_from_checkpoint(self, resume_dict):

        if resume_dict is not None:
            print('==> Resuming from checkpoint: ',resume_dict)
            if self.args.training_strategy == "LP+FT":
                model_path = os.path.join(self.LP_path, resume_dict+'.pth')
            else :
                model_path = os.path.join(self.path_cp, resume_dict+'.pth')
            checkpoint = torch.load(model_path, map_location=device)
            # if self.args.training_strategy == "LP+FT":
            #     self.start_epoch = 0
            #     self.last_chkpt_name = 'init'
            # else :
            #     self.start_epoch = checkpoint['epoch']+1
            #     self.last_chkpt_name = resume_dict
            #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.best_map = checkpoint['best_map']
            
    def build_model(self):
        cfg = self.args
        classnames = self.tr_classes

        print(f"Loading CLIP (backbone: {cfg.clip_backbone})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(device)
        # CLIP's default precision is fp16
        clip_model.float()
        
        print("Building custom CLIP")
        
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        # train_parameters = ['image_encoder.prompt_embeddings', 'image_encoder.proj', 'prompt_learner.ctx', 'text_encoder.text_projection', "image_encoder.deep_prompt_embeddings"]
        train_parameters = ['image_encoder.proj', 'image_encoder.prompt_embeddings', 'prompt_learner.ctx', 'text_encoder.text_projection', "image_encoder.deep_prompt_embeddings"]
        tot = 0
        train_part = 0
        for name, param in self.model.named_parameters():
            tot += param.numel()
            print(name)
            if name not in train_parameters:
                param.requires_grad_(False)
        # self.model.image_encoder.proj.requires_grad_(True)
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        print("=============train==============")
        for name, param in self.model.named_parameters():
            # print(name)
            if param.requires_grad == True:
                train_part += param.numel()
                print(name)
        print(f"tot={tot}, train = {train_part}")
        self.model.to(device)
        # NOTE: only give prompt_learner to the optimizer
        if self.args.optimizer=='sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), weight_decay=self.args.l2_reg, momentum=self.args.momentum, nesterov=False, lr=self.args.lr)
        elif self.args.optimizer=='adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.l2_reg)

        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
       #  self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

@torch.no_grad()
def evaluate(loader_sketch, loader_image, model:CustomCLIP, dict_clss):

    # Switch to test mode
    model.eval()

    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in enumerate(loader_sketch):

        sk = sk.float().to(device)

        # Sketch embedding into a semantic space
        sk_em = model.image_encoder(sk)
        # sk_em = model.base_model.last_linear(sk_feat)
        # Accumulate sketch embedding
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(utils.numeric_classes(cls_sk, dict_clss)).long().to(device)

        sketchLabels.append(cls_numeric)
        # if i == 2 :
        #     break

    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in enumerate(loader_image):

        im = im.float().to(device)

        # Image embedding into a semantic space
        im_em = model.image_encoder(im)
        # im_em = model.base_model.last_linear(im_feat)

        # Accumulate sketch embedding
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().to(device)

        realLabels.append(cls_numeric)
        # if i == 2 :
        #     break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    return eval_data
