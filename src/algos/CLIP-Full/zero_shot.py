"""
Zero-Shot CLIP, using the CLIP model from OpenAI's CLIP paper, and the zero-shot retrieval evaluation protocol.
"""
import os
import sys
code_path = 'INSERT_CODE_PATH_HERE' # e.g. '/home/username/ProS' 
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.DomainNet import domainnet
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import clip
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.dataloaders import CuMixloader, BaselineDataset
from data.sampler import BalancedSampler
from utils import utils, GPUmanager
from ProS.src.options.options_clip import Options
from utils.logger import AverageMeter
from utils.metrics import compute_retrieval_metrics
from tqdm import tqdm

gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

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
        data_splits = data_input['splits']

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        self.CLIP, self.preprocess = clip.load('ViT-B/32', device, jit=False)
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
        self.dict_clss = utils.create_dict_texts(self.tr_classes)  # 生成 类:index 的一个字典
        self.dict_te_clss = utils.create_dict_texts(self.te_classes+self.tr_classes)

        fls_tr = data_splits['tr']
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

        data_va_query = BaselineDataset(data_splits['query_va'], transforms=self.image_transforms['eval'])
        data_va_gallery = BaselineDataset(data_splits['gallery_va'], transforms=self.image_transforms['eval'])


        # PyTorch valid loader for query
        self.va_loader_query = DataLoader(dataset=data_va_query, batch_size=args.batch_size * 5, shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
        # PyTorch valid loader for gallery
        self.va_loader_gallery = DataLoader(dataset=data_va_gallery, batch_size=args.batch_size * 5, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

        print(
            f'#Tr samples:{len(data_train)}; #Val queries:{len(data_va_query)}; #Val gallery samples:{len(data_va_gallery)}.\n')
        print('Loading Done\n')


    
    @torch.no_grad()
    def evaluate(self, loader_query, loader_gallery):
        self.CLIP.eval()
        # Start counting time
        time_start = time.time()
        queryEmbedding = list()
        queryLabels = list()
        for i, (im, label, dom) in tqdm(enumerate(loader_query), desc='Extrac query feature', total=len(loader_query)):
            # for idx, image in enumerate(im):
            #     im[idx] = self.preprocess(image).unsqueeze(0)
            im = im.to(device)
            # print(im.device)
            im_feat = self.CLIP.encode_image(im)
            queryEmbedding.append(im_feat)
            label_numeric = torch.from_numpy(utils.numeric_classes(label, self.dict_te_clss)).long().to(device)
            queryLabels.append(label_numeric)
        queryLabels = torch.cat(queryLabels, 0)
        queryEmbedding = torch.cat(queryEmbedding, 0)

        galleryEmbedding = list()
        galleryLabels = list()
        for i, (im, label, dom) in tqdm(enumerate(loader_gallery), desc='Extrac gallery feature', total=len(loader_gallery)):
            # for idx, image in enumerate(im):
            #     im[idx] = self.preprocess(image).unsqueeze(0)
            im = im.to(device)
            im_feat = self.CLIP.encode_image(im)
            galleryEmbedding.append(im_feat)
            label_numeric = torch.from_numpy(utils.numeric_classes(label, self.dict_te_clss)).long().to(device)
            galleryLabels.append(label_numeric)
        galleryLabels = torch.cat(galleryLabels, 0)
        galleryEmbedding = torch.cat(galleryEmbedding, 0)
        run_time = time.time() - time_start
        # print('\nTime:{}; Query Emb Dim:{}; Gallery Emb Dim:{}'.format(run_time, queryEmbedding.shape, galleryEmbedding.shape))
        eval_data = compute_retrieval_metrics(queryEmbedding, queryLabels, galleryEmbedding, galleryLabels)
        eval_data['time'] = run_time
        return eval_data

    def test(self):
        # outstr = '-'*10 + "  -hd: " + args.holdout_domain + "  -sd: " + args.seen_domain + "  " + '-'*10 + '\n\n'
        for domain in ['quickdraw', 'clipart', 'sketch', 'painting', 'infograph']:
            for gzs in [0, 1]:
                test_head_str = 'Query:' + domain + '; Gallery:' + args.gallery_domain + '; Generalized:' + str(gzs)
                print(test_head_str)

                splits_query = domainnet.trvalte_per_domain(args, domain, 0, self.tr_classes, self.va_classes, self.te_classes)
                splits_gallery = domainnet.trvalte_per_domain(args, args.gallery_domain, gzs, self.tr_classes, self.va_classes,
                                                              self.te_classes)

                data_te_query = BaselineDataset(splits_query['te'], transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(splits_gallery['te'], transforms=self.image_transforms['eval'])

                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=64 * 5, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=64 * 5, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)

                print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')
                te_data = self.evaluate(te_loader_query, te_loader_gallery)

                out =  "\nTime = %.4f, mAP@200 = %.4f, Prec@200 = %.4f\n" % (te_data['time'], te_data['mAP@200'], te_data['prec@200'])
                print(out)

def main(args):
    use_gpu = torch.cuda.is_available()
    # print('\nDevice:{}'.format(device))

    trainer = Trainer(args)
    trainer.test()


if __name__ == '__main__':
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)
