
import json
import sys
import os
code_path = 'INSERT_CODE_PATH_HERE' # e.g. '/home/username/ProS' 
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.DomainNet import domainnet
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import clip
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.dataloaders import  BaselineDataset
from utils import utils, GPUmanager
from options.options_clip import Options
from utils.metrics import compute_retrieval_metrics

# user defined
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

class Test:

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
        data_splits = data_input['splits']
        self.data_splits = data_splits
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        self.CLIP, self.preprocess = clip.load(args.clip_backbone, device, jit=False)
        self.CLIP = self.CLIP.float().eval()
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

            'eval': self.preprocess
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes) 
        self.te_dict_class = utils.create_dict_texts(self.tr_classes+self.va_classes+self.te_classes)

        fls_tr = data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)
        print(self.dict_doms)
        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)

        
        self.model = self.CLIP
       
        self.text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.tr_classes]).to(device)
        self.RG = np.random.default_rng()
       
       
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


        self.suffix = 'es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
                        '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)+'_seed-'+str(args.seed)

        # exit(0)
        self.path_cp = os.path.join(args.code_path, '/src/algos/Baseline_CLIP/saved_models', args.training_strategy, args.dataset, save_folder_name)
        
        print('Done\n')

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_chkpt_name='init'

        self.resume_from_checkpoint(args.resume_dict)
        

    def do_testing(self):

        print('***Train***')
       
            # evaluate on validation set, map_ since map is already there
        print('\n***Validation***')
        te_data = []
        if self.args.dataset=='DomainNet':
            if self.args.ucddr == 0:
                te_data = []
                for domain in [self.args.holdout_domain]:
                    for includeSeenClassinTestGallery in [1]:
                        test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(includeSeenClassinTestGallery)
                        print(test_head_str)
                        
                        splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes, self.va_classes, self.te_classes)
                        splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain, includeSeenClassinTestGallery, self.tr_classes, self.va_classes, self.te_classes)

                        data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=self.image_transforms['eval'])
                        data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=self.image_transforms['eval'])

                        # PyTorch test loader for query
                        te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)
                        # PyTorch test loader for gallery
                        te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)

                        # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                        result = evaluate(te_loader_query, te_loader_gallery, self.model,self.te_dict_class, self.args)
                        te_data.append(result)
                        out ="mAP@200 = %.4f, Prec@200 = %.4f\n"%(result['mAP@200'], result['prec@200'])
                        print(out)
                        class_ap_dict = result["class_ap"]
    
                        info = json.dumps(class_ap_dict,indent=4,separators=(',',': '))
                        f = open('./ip_UCDR_class_ap.json', 'w')
                        f.write(info)

                        print(out)
                       
                    
                map_ = te_data[0][self.map_metric]
                prec = te_data[0][self.prec_metric]
            else :

                if self.args.holdout_domain == 'quickdraw':
                    p = 0.1
                else :
                    p = 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)

                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])

                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=2048, shuffle=False,
                                                num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=2048, shuffle=False,
                                                num_workers=self.args.num_workers, pin_memory=True)

                # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.args)
                map_ = te_data[self.map_metric]
                prec = te_data[self.prec_metric]
                out ="mAP@200 = %.4f, Prec@200 = %.4f\n"%(map_, prec)
                print(out)
        else :

            data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
            data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

            te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
            te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

            print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

            te_data = evaluate(te_loader_query, te_loader_gallery, self.model,self.te_dict_class, self.args)
            print(te_data)
            map_ = te_data[self.map_metric]
            prec = te_data[self.prec_metric]
        # prec = valid_data[self.prec_metric]
            out ="mAP@200 = %.4f, Prec@200 = %.4f\n"%(map_, prec)
            print(out)

    
    def resume_from_checkpoint(self, resume_dict):

        if resume_dict is not None:
            print('==> Resuming from checkpoint: ',resume_dict)
            model_path = os.path.join(self.path_cp, resume_dict+'.pth')
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)


@torch.no_grad()
def evaluate(loader_sketch, loader_image, model, dict_clss, args):

    # Switch to test mode
    model.eval()

    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in enumerate(loader_sketch):

        sk = sk.float().to(device)

        # Sketch embedding into a semantic space
        sk_em = model.encode_image(sk)
        # sk_em = model.base_model.last_linear(sk_feat)
        # Accumulate sketch embedding
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(utils.numeric_classes(cls_sk, dict_clss)).long().to(device)

        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break

    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in enumerate(loader_image):

        im = im.float().to(device)

        # Image embedding into a semantic space
        im_em = model.encode_image(im)
        # im_em = model.base_model.last_linear(im_feat)

        # Accumulate sketch embedding
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    return eval_data



def main(args):

    Test(args).do_testing()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)
