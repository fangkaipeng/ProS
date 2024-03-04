import sys
import os
from tqdm import tqdm
import os 
code_path = 'INSERT_CODE_PATH_HERE' # e.g. '/home/username/ProS' 
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
sys.path.append(os.path.join(code_path, "clip"))
from models.prosnet import prosnet
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.DomainNet import domainnet
import numpy as np
import torch.backends.cudnn as cudnn
from data.dataloaders import BaselineDataset
from utils import utils, GPUmanager
from utils.metrics import compute_retrieval_metrics
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

from options.options_pros import Options



class Tester:

    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)

        self.tr_classes = data_input['tr_classes']
        self.va_classes = data_input['va_classes']
        self.te_classes = data_input['te_classes']
        data_splits = data_input['splits']
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Image transformations
        self.image_transforms = {
            'eval': transforms.Compose([
                transforms.Resize(args.image_size, interpolation=BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                #lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes) 
        self.te_dict_class = utils.create_dict_texts(self.tr_classes+self.va_classes+self.te_classes)

        fls_tr = data_splits['tr']
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)
        print(self.dict_doms)

        if args.dataset=='DomainNet':
            self.save_folder_name = 'seen-'+args.seen_domain+'_unseen-'+args.holdout_domain+'_x_'+args.gallery_domain
        
        self.path_cp = os.path.join(args.code_path,'saved_models', args.dataset, self.save_folder_name)
        self.model = prosnet(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

        if args.dataset=='DomainNet' or (args.dataset=='Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

        self.resume_from_checkpoint(args.resume_dict)


    def test(self):    
        te_data = []
        if self.args.dataset == 'DomainNet':
            if self.args.udcdr == 0:
                for domain in [self.args.seen_domain, self.args.holdout_domain]:
                    for includeSeenClassinTestGallery in [0,1]:
                        test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(includeSeenClassinTestGallery)
                        print(test_head_str)
                        
                        splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes, self.va_classes, self.te_classes)
                        splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain, includeSeenClassinTestGallery, self.tr_classes, self.va_classes, self.te_classes)

                        data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=self.image_transforms['eval'])
                        data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=self.image_transforms['eval'])

                        # PyTorch test loader for query
                        te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)
                        # PyTorch test loader for gallery
                        te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)

                        # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                        result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4, self.args)
                        te_data.append(result)
                        
                        out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(result[self.map_metric], result[self.prec_metric])
                        print(out)
            else:
               
                if self.args.holdout_domain == 'quickdraw':
                    p = 0.1
                else :
                    p = 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)

                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])

                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=800, shuffle=False,
                                                num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=800, shuffle=False,
                                                num_workers=self.args.num_workers, pin_memory=True)
                result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4, self.args)
                map_ = result[self.map_metric]
                prec = result[self.prec_metric]
                out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(map_, prec)
                print(out)
        else :
            data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
            data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

            te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
            te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

            print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

            te_data = evaluate(te_loader_query, te_loader_gallery, self.model,self.te_dict_class, self.dict_doms, 4, self.args)
            out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(te_data[self.map_metric], te_data[self.prec_metric])
        
            map_ = te_data[self.map_metric]
            prec = te_data[self.prec_metric]
            out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(map_, prec)
            print(out)

    def resume_from_checkpoint(self, resume_dict):
        if resume_dict is not None:
            print('==> Resuming from checkpoint: ',resume_dict)
            model_path = os.path.join(self.path_cp, resume_dict+'.pth')
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

@torch.no_grad()
def evaluate(loader_sketch, loader_image, model:prosnet, dict_clss, dict_doms, stage, args):

    # Switch to test mode
    model.eval()

    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)
        sk_em = model.image_encoder(sk, dom_id, cls_id, stage)
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if  args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)  
        # Sketch embedding into a semantic space
        im_em = model.image_encoder(im, dom_id, cls_id, stage)
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2 :
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    return eval_data



def main(args):
    trainer = Tester(args)
    trainer.test()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)