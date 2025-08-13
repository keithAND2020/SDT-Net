import pdb
# pdb.set_trace()
import argparse
import torch
import os
import sys

import io
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from libs.builders import build_trainer, build_tester, build_models, build_dataloaders
from libs.utils import init_all
from torch import autograd
def main():
    parser = argparse.ArgumentParser(description='3D dark matter reconstruction')
    parser.add_argument('config', type=str)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--launcher', choices=['none', 'slurm', 'pytorch'], default='none')
    args = parser.parse_args()

    configs, logger = init_all(args)
    
    trainloader, evalloader = build_dataloaders(**configs['dataset'])
    
    net = build_models(**configs['model']).to('cuda')
    if args.evaluate:
        tester = build_tester(net,
                          evalloader,
                          logger,
                          **configs['test']
                     )
        '''
        for i in range(50):
            tester.model.threshold = (i+1)*0.5
            res_string,_ = tester.eval()
            print('%f:'%tester.model.threshold)
            print(res_string)
        '''
        tester.eval()
        # res_string,_ = tester.eval()
        # print(res_string)
        exit()
    with autograd.detect_anomaly():
        trainer = build_trainer(net, 
                                logger,
                                trainloader, 
                                evalloader,
                                # ddp=True,
                                **configs['train'])
        trainer.train()
        

if __name__ == "__main__":
    main()