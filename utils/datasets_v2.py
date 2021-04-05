from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def load_npz(path) -> dict:
    sample = np.load(path, allow_pickle=True)
    ret = {key: sample[key] for key in sample}

    return ret

# class ToTensor(object):
#     """ 샘플 안에 있는 n차원 배열을 Tensor로 변홥힙니다. """
#     #https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
#     def __call__(self, sample):
#         beta, bonelength = sample['beta'], sample['bonelength']
#         return {'beta': torch.FloatTensor(beta),
#                 'bonelength': torch.FloatTensor(bonelength)}

#https://tutorials.pytorch.kr/recipes/recipes/custom_dataset_transforms_loader.html
"""
female max bonelength
0.27402371,0.26814702,0.25418773,0.65134838,0.64194265,0.33568995,0.74821297,0.74431565,0.23773826,0.28092859,0.2713996,0.42482841,0.32642692,0.32159231,0.2429592,0.22996855,0.21920209,0.44671307,0.44182389,0.43590134,0.45621939,0.17261486,0.17406494
normalize = 1/0.374289 = 2.67173
mean = 0.373924
x,y,z
max, min = 0.035514124332456064, -0.04808932240940057
"""
#TODO: change normalization(is it okay to use from max between x,y and z to normalize
#       all x,y and z, not each max to normalize each axis?) or standardization
class Normalize(object):
    """ 주어진 크기로 샘플안에 있는 이미지를 재변환 합니다.

    Args:
        output_size (tuple 또는 int): 원하는 결과값의 크기입니다.
        tuple로 주어진다면 결과값은 output_size 와 동일해야하며
        int일때는 설정된 값보다 작은 이미지들의 가로와 세로는 output_size 에 적절한 비율로 변환됩니다.
        beta_val (:np.array:np.float32) : normalized beta_val is same with normalized delta_beta_shape
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        beta_val, bonelength_val = sample['beta'], sample['bonelength']
        # beta_val = (beta_val-0.0)/0.048089/5
        beta_val /= 1.35

        # bonelength_val = (bonelength_val - 0.373924) * 2.67173 * 0.5 + 0.5
        bonelength_val = (bonelength_val - 0.373924) * 2.67173

        return {'beta': beta_val, 'bonelength': bonelength_val}

class StarBetaBoneLengthDataset(Dataset):
    def __init__(self, path, device, transform=None, debug=-1):
        self.data = load_npz(path=path)
        self.transform = transform
        self.device = device
        self.debug = debug
        # assert (True, "Data type is incorrect('train', 'test', 'validation') or param. datasize_dict is incorrect.")

    def __len__(self):
        # return self.data['beta'].shape[0]
        return 2

    def __getitem__(self, idx):
        if self.debug is not -1:
            idx = self.debug
        ret = {key: torch.tensor(self.data[key][idx], dtype=torch.float32, device=self.device) for key in self.data}

        #
        if self.transform:
            ret = self.transform(ret)

        return ret['beta'], ret['bonelength']