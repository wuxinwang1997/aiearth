# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import argparse
import os
import sys
sys.path.append('.')
from os import mkdir
from config import cfg
from data import make_test_data_loader
from engine.predicter import Predicter
from modeling import build_model
import random
import torch
import numpy as np
from utils.modelema import ModelEMA

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def predict(cfg):
    seed_everything(cfg.SEED)
    model = build_model(cfg)
    ema = ModelEMA(model)
    if torch.cuda.is_available():
        device = 'cuda'
        ema.ema.load_state_dict(torch.load(cfg.TEST.WEIGHT)['model_state_dict'])
    else:
        device = 'cpu'
        ema.ema.load_state_dict(torch.load(cfg.TEST.WEIGHT, map_location=device)['model_state_dict'])

    test_loader = make_test_data_loader(cfg)

    predicter = Predicter(model=ema.ema, device=device, cfg=cfg, test_loader=test_loader)
    predicter.predict()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    result_dir = cfg.RESULT_DIR
    if result_dir and not os.path.exists(result_dir):
        mkdir(result_dir)
    predict(cfg)


if __name__ == '__main__':
    main()
