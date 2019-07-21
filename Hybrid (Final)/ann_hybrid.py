#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:04:01 2019

@author: cyberhunters
"""


import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(134, 100),nn.LeakyReLU(True))
        
        self.feature1 = nn.Sequential(nn.Linear(100, 9),nn.LeakyReLU(True))
        
        
    def forward(self, x):
        x = self.feature(x)

        x = self.feature1(x)
        return x

