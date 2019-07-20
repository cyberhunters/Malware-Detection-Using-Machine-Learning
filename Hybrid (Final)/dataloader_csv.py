#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:15:29 2019

@author: aqsas
"""

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
class CustomDatasetFromImages(Dataset):
    
    def __init__(self, csv_path):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0:-2])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, -1])-1
        
        
        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, -2])
        # Calculate len
        self.data_len = len(self.data_info.index)
        
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        single_image_name = np.array(single_image_name)
        single_image_name = np.resize(single_image_name,(32,32))
        img_as_tensor = torch.Tensor(single_image_name)
        img_as_tensor = torch.unsqueeze(img_as_tensor,0)
        single_image_label = self.label_arr[index]
        name=self.operation_arr[index]
        return (img_as_tensor,name,single_image_label)
        

    def __len__(self):
        return self.data_len    
    

