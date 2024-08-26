# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      ts2img
   Author :         zmfy
   DateTime :       2024/5/17 20:41
   Description :    
-------------------------------------------------
"""
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField
from sklearn import preprocessing


def ts2img_by_GASF(data, img_size=224):
    gaf = GramianAngularField(image_size=img_size, method='summation')
    images = gaf.fit_transform(data)
    return images


def ts2img_by_GADF(data, img_size=224):
    gaf = GramianAngularField(image_size=img_size, method='difference')
    images = gaf.fit_transform(data)
    return images


def ts2img_by_MarkovTF(data, img_size=224):
    mtf = MarkovTransitionField(image_size=img_size)
    images = mtf.fit_transform(data)
    return images


def ts2img_factor(data, img_size, trans_method, need_scale=False, **kwargs):
    """
    批量转换时序数据到图像
    :param data: 二维时序数据，【样本数，时序维度】, data的每一行将被转换为图像
    :param img_size:图像尺寸
    :param trans_method: 转换方法，可选'GASF'(默认）, 'GADF', 'Markov',
    :param need_scale: 否需要0-1标准化
    :param kwargs:
    :return:
    """
    if isinstance(data, list):
        data = np.array(data)
    if need_scale:
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data.T)
        data = data.T

    if trans_method == 'GASF':
        return ts2img_by_GASF(data, img_size)
    elif trans_method == 'GADF':
        return ts2img_by_GADF(data, img_size)
    elif trans_method == 'Markov':
        return ts2img_by_MarkovTF(data, img_size)
    else:
        raise ValueError
