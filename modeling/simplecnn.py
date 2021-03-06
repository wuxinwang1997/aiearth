# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        if cfg.MODEL.BACKBONE.PRETRAIN:
           self.model.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_PATH))
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, 24)
        #self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.model.fc(x)
        return x


