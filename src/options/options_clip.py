"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='CLIP-Full for UCDR/ZS-SBIR')
        parser.add_argument('-debug_mode', '--debug_mode',default=1, type=int, help='use debug model')
        parser.add_argument('-udcdr', '--udcdr',default=0, type=int, help='evaluate udcdr?')
        parser.add_argument('-code_path', '--code_path', default='', type=str, help='code path of ProS')
        parser.add_argument('-dataset_path', '--dataset_path', default='', type=str, help='Path of three datasets')
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')
        
        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='painting', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='clipart', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'], help='chose unseen domain')
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\
                            domains other than seen domain and gallery')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')

        # Loss weight & reg. parameters
        parser.add_argument('-alpha', '--alpha', default=1.0, type=float, help='Parameter to scale weights for Class Similarity Matrix')
        parser.add_argument('-l2', '--l2_reg', default=0.00004, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-semsz', '--semantic_emb_size', choices=[200, 300], default=300, type=int, help='Glove vector dimension')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        
        # Model parameters
        parser.add_argument('-clip_backbone', '--clip_backbone', type=str, choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32', help='choose clip backbone')
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=156, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=8, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=15, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=100, metavar='N', help='How many batches to wait before logging training status')

        parser.add_argument('-trainvalid', '--trainvalid', choices=[0, 1], default=1, type=int)

        parser.add_argument('-ac_grad', '--ac_grad', default=16, type=int)

        self.parser = parser

    
    def parse(self):
        # Parse the arguments test
        return self.parser.parse_args()