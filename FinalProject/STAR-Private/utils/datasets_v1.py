from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#https://tutorials.pytorch.kr/recipes/recipes/custom_dataset_transforms_loader.html
class ToTensor(object):
    """ 샘플 안에 있는 n차원 배열을 Tensor로 변홥힙니다. """
    #https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
    def __call__(self, sample):
        beta, bonelength = sample['beta'], sample['bonelength']
        return {'beta': torch.FloatTensor(beta),
                'bonelength': torch.FloatTensor(bonelength)}

class Normalize(object):
    """ 주어진 크기로 샘플안에 있는 이미지를 재변환 합니다.

    Args:
        output_size (tuple 또는 int): 원하는 결과값의 크기입니다.
        tuple로 주어진다면 결과값은 output_size 와 동일해야하며
        int일때는 설정된 값보다 작은 이미지들의 가로와 세로는 output_size 에 적절한 비율로 변환됩니다.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def __call__(self, sample):
        beta_val, bonelength_val = sample['beta'], sample['bonelength']
        beta_val = (beta_val-self.mean)/(self.std * 2.0) + 0.5
        # print(beta_val)
        minElement = np.amin(beta_val)
        maxElement = np.amax(beta_val)
        #print(f'{minElement},  {maxElement}')

        return {'beta': beta_val, 'bonelength': bonelength_val}

class StarBetaBoneLengthDataset(Dataset):
    def __init__(self, npy_file, transform=None, length=None, value=None):
        if npy_file is not None:
            self.beta_and_jointposition = np.load(npy_file).astype(np.float32)
        elif npy_file is None and value is not None:
            self.beta_and_jointposition = value.astype(np.float32)

        self.transform = transform
        if length is None or length < 0 or length > len(self.beta_and_jointposition):
            self.len = len(self.beta_and_jointposition)
        else:
            self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'beta':self.beta_and_jointposition[idx,:300],'bonelength':self.beta_and_jointposition[idx,300:]}

        if self.transform:
            sample = self.transform(sample)
        #print(f'len{sample["bonelength"].shape}')

        return (sample['beta'], sample['bonelength'])
