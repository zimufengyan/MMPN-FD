# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      ts2img
   Author :         zmfy
   DateTime :       2024/5/17 20:41
   Description :    
-------------------------------------------------
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from PIL import Image
from matplotlib import cm
from pyts.image import GramianAngularField, MarkovTransitionField
from scipy.signal import stft
from sklearn import preprocessing


def arr2image(arr):
    sm = cm.ScalarMappable()
    img = sm.to_rgba(arr, bytes=True)
    pil_shape = (img.shape[1], img.shape[0])
    img = Image.frombuffer(
        "RGBA", pil_shape, img, "raw", "RGBA", 0, 1)
    img = img.convert("RGB")
    return img


def ts2img_by_GASF(data, img_size=224):
    gaf = GramianAngularField(image_size=img_size, method='summation')
    arr = gaf.fit_transform(data)[0]
    img = arr2image(arr)
    return img


def ts2img_by_GADF(data, img_size=224):
    gaf = GramianAngularField(image_size=img_size, method='difference')
    arr = gaf.fit_transform(data)[0]
    img = arr2image(arr)
    return img


def ts2img_by_MarkovTF(data, img_size=224):
    mtf = MarkovTransitionField(image_size=img_size)
    arr = mtf.fit_transform(data)[0]
    img = arr2image(arr)
    return img


def ts2img_factor(data, img_size=224, trans_method='GASF', need_scale=False):
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


def stft_trans(data, fs=25000, window='hann', window_size=32, overlap=None):
    # data shape: [bsize, seq_len]
    overlap = window_size // 2 if overlap is None else overlap
    f, t, zxx = stft(data, fs=fs, window=window, nperseg=window_size, noverlap=overlap)
    result = np.transpose(np.abs(zxx), [0, 2, 1])
    return result  # shape: [bsize, (1 + N // (nperseg - noverlap), 1 + nfft // 2)]


def get_time_freq_image(data, fs=25000, window='hann', window_size=32, overlap=None):
    overlap = window_size // 2 if overlap is None else overlap
    f, t, zxx = stft(data, fs=fs, window=window, nperseg=window_size, noverlap=overlap)
    plt.figure(figsize=(6, 6), dpi=100)
    fig, ax = plt.subplots()
    ax.pcolormesh(t, f, np.abs(zxx), shading='gouraud')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    img = img.convert('RGB')
    return img


def sliding_window(data, window_size, overlap):
    num_windows = (len(data) - window_size) // (window_size - overlap) + 1
    windows = np.zeros((num_windows, window_size))  # 创建存储子序列的数组

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        windows[i] = data[start:end]

    return windows


def preprocess_data(data, length, max_samples, overlap=100):
    data = np.reshape(data, (-1))  # shape [n]
    data = sliding_window(data, length, overlap)
    data = data[:max_samples, :]

    return data


def open_data(bath_path, key_num, name="_DE_time"):
    # name should in ["_DE_time", "_FE_time"]
    path = os.path.join(bath_path, str(key_num) + '.mat')
    str1 = "X" + "%03d" % key_num + name
    data = scio.loadmat(path)
    data1 = data[str1]  # shape: [n, 1]
    return data1


if __name__ == '__main__':
    # x = np.random.randn(512).reshape(1, -1)
    # print(stft_trans(x).shape)
    x = np.array(range(0, 1000))
    y = sliding_window(x, 100, 10)
    print(y.shape)
