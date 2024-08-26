# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      utis
   Author :         zmfy
   DateTime :       2024/5/17 19:40
   Description :    
-------------------------------------------------
"""
import numpy as np
from scipy.signal import stft
from scipy.fftpack import fft
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import json


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, object):
            return obj.__dict__
        else:
            return super(MyJsonEncoder, self).default(obj)


def set_seed(seed):
    if seed == -1:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_similarity(x, y, method="euclidean"):
    """Compute the similarity between the x and y"""
    if x.size(1) != y.size(1):
        raise RuntimeError(
            f"The size of tensor text ({x.size(1)}) must match the size of tensor y ({y.size(1)}) at dimension 1")
    n, m = x.size(0), y.size(0)
    d = x.size(1)

    a = x.unsqueeze(1).expand(n, m, d)
    b = y.unsqueeze(0).expand(n, m, d)

    if method == 'euclidean':
        return -torch.pow((a - b), 2).sum(2)
    elif method == 'cosine':
        return F.cosine_similarity(a, b, dim=2)
    else:
        raise ValueError(f"unsupported similarity function ({method})")


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'delta_weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'one-zero':
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'delta_betas') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's delta_weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def fft_trans(data, fs):
    """
    对输入信号进行FFT
    :param fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result


def set_logger(log_path, log_name):
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(log_path, '{}.log'.format(log_name)),
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def binary_accuracy_with_logits(output: torch.Tensor, target: torch.Tensor):
    """
    计算二分类准确率
    :param output: Tensor, size of (batch_size), output of model without sigmoid
    :param target: Tensor, size of (batch_size), ground truth label, 0 or 1
    :return: tensor
    """
    logits = torch.sigmoid(output)
    preds = torch.where(logits > 0.5, torch.ones_like(logits), torch.zeros_like(logits)).long()
    correct = preds.eq(target.long())
    accuracy = correct.sum() / target.size(0)
    return accuracy


if __name__ == '__main__':
    data = np.random.randn(1000)
