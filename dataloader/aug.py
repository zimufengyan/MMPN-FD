# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      aug
   Author :         zmfy
   DateTime :       2024/5/27 12:02
   Description :    
-------------------------------------------------
"""
import torch


class Reshape(object):
    def __call__(self, x: torch.Tensor):
        dim = torch.prod(torch.tensor(x.size())).item()
        return x.reshape(dim, 1)


class Normalize(object):
    def __init__(self, method="0-1"):  # "0-1","-1-1","mean-std"
        self.method = method

    def __call__(self, x: torch.Tensor):
        if self.method == "0-1":
            x = (x - x.min()) / (x.max() - x.min())
        elif self.method == "-1-1":
            x = 2 * (x - x.min()) / (x.max() - x.min()) + -1
        elif self.method == "mean-std":
            x = (x - x.mean()) / x.std()
        else:
            raise NameError('This normalization is not included!')
        return x
