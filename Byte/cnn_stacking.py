#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:59:45 2019

@author: aqsas
"""



import torch.nn as nn

class Cnn_Stacking(nn.Module):
    def __init__(self,mod,mod1):
        super(Cnn_Stacking, self).__init__()
        self.feature = mod.encoder
        self.feature1 = mod1.encoder
        self.feature2 = nn.Sequential(nn.Linear(256*8 * 8, 500),nn.ReLU(True))
        self.feature3 = nn.Sequential(nn.Linear(500, 9),nn.ReLU(True))
        
    def forward(self, x):
        x = self.feature(x)
        x = self.feature1(x)
        x = x.reshape(x.size(0), -1)
        x = self.feature2(x)
        x = self.feature3(x)
#        x = self.classifier(x)
        return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)  # b, 16, 5, 5
       )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autoencoder1(nn.Module):
    def __init__(self):
        super(autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)  # b, 16, 5, 5
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

