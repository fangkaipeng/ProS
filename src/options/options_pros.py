"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='ProS for UCDR/ZS-SBIR')


        # temp
        parser.add_argument('-generator_layer', '--generator_layer',default=2, type=int, help='projection for domain visual prompt')
        parser.add_argument('-stage1_epochs', '--stage1_epochs',default=1, type=int, help='projection for domain visual prompt')
        
        
       
        parser.add_argument('-code_path', '--code_path', default='', type=str, help='code path of ProS')
        parser.add_argument('-dataset_path', '--dataset_path', default='', type=str, help='Path of three datasets')
        parser.add_argument('-resume', '--resume_dict', type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')
        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str, choices=['RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32', help='choose clip backbone')
        parser.add_argument('-CLS_NUM_TOKENS', '--CLS_NUM_TOKENS',default=300, type=int, help='number of Semantic Prompt Units, usually equals to the number of classes')
        parser.add_argument('-DOM_NUM_TOKENS', '--DOM_NUM_TOKENS',default=5, type=int, help='number of Domain Prompt Units, usually equals to the number of domains')
        parser.add_argument('-DOM_PROJECT', '--DOM_PROJECT',default=-1, type=int, help='projection for Domain Prompt Units')
        parser.add_argument('-CLS_PROJECT', '--CLS_PROJECT',default=-1, type=int, help='projection for Semantic Prompt Units')
        parser.add_argument('-VP_INITIATION', '--VP_INITIATION',default='random', type=str, help='initiation for visual prompt')
        parser.add_argument('-GP_DOM_NUM_TOKENS', '--GP_DOM_NUM_TOKENS',default=1, type=int, help='Prompt Template for Domain Prompt')
        parser.add_argument('-GP_CLS_NUM_TOKENS', '--GP_CLS_NUM_TOKENS',default=1, type=int, help='Prompt Template for Semantic Prompt')
        parser.add_argument('-tp_N_CTX', '--tp_N_CTX',default=16, type=int, help='text prompt length')
        parser.add_argument('-use_NTP', '--use_NTP', default=0, type=int, help='use normal text prompt tuning (tuning all prompt)')
        parser.add_argument('-debug_mode', '--debug_mode',default=1, type=int, help='use debug model')
        parser.add_argument('-dropout', '--dropout',default=0.5, type=float, help='dropout rate')

        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='painting', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='infograph', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\domains other than seen domain and gallery')
        parser.add_argument('-udcdr', '--udcdr', choices=[0, 1], default=0, type=int, help='whether or not to evaluate under udcdr protocol')


        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')

        # Loss weight & reg. parameters
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=10, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')

        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=200, metavar='N', help='Number of epochs to train in stage 2')
        parser.add_argument('-lr', '--lr', type=float, default=0.0001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=2, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser


    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
