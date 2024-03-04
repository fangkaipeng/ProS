import sys
import torch
import os 
code_path = 'INSERT_CODE_PATH_HERE' # e.g. '/home/username/ProS' 
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))

# user defined

from trainer import Trainer
from options.options_pros import Options

def main(args):
    trainer = Trainer(args)
    trainer.do_training()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)