"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='PromptTuning for UCDR/ZS-SBIR')

       
        parser.add_argument('-code_path', '--code_path', default='', type=str, help='code path of ProS')
        parser.add_argument('-dataset_path', '--dataset_path', default='', type=str, help='Path of three datasets')
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')
        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str, choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32', help='choose clip backbone')
        parser.add_argument('-tp_N_CTX', '--tp_N_CTX', type=int, default=16, help='number of text prompt context tokens')
        parser.add_argument('-tp_CTX_INIT', '--tp_CTX_INIT', type=str, default="a photo of a", help='context tokens init')
        parser.add_argument('-ts', '--training_strategy',default='VP', type=str, choices=['TP', 'VP'], help='training_strategy,TP is CoOp, VP is VPT in the paper')
        parser.add_argument('-vp_NUM_TOKENS', '--vp_NUM_TOKENS',default=10, type=int, help='number of visual prompt tokens')
        parser.add_argument('-vp_PROJECT', '--vp_PROJECT',default=-1, type=int, help='projection for visual prompt')
        parser.add_argument('-vp_INITIATION', '--vp_INITIATION',default='random', type=str, help='initiation for visual prompt')
        parser.add_argument('-vp_DEEP', '--vp_DEEP',default=False, type=bool, help='deep viusla prompt')
        parser.add_argument('-debug_mode', '--debug_mode',default=1, type=int, help='use debug model, program will break down after a few iterations.')
       
        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='infograpg', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='painting', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\
                            domains other than seen domain and gallery')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')

        # Loss weight & reg. parameters
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=65, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')

        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser


    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
