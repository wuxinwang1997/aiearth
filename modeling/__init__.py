# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .simplecnn import SimpleCNN
def build_model(cfg):
    model = SimpleCNN(cfg)
    return model
