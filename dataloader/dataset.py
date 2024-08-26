# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      dataset
   Author :         zmfy
   DateTime :       2024/5/17 16:15
   Description :    
-------------------------------------------------
"""
import os
from typing import Sequence, Union

import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from matplotlib import image
from matplotlib import cm

from dataloader.tools import *
from dataloader import aug
from utils import *


FE_DATA_DICT = {
    'IF': {
        '0007': 278,    # 278-281 for hp 0-4
        '0014': 274,    # 274-277 for hp 0-4
        '0021': 270,    # 270-273 for hp 0-4
    },
    'OF': {
        '0007': 298,    # 298-301
        '0014': 309,    # 309-312 for hp 1 0 2 3
        '0021': 315,    # 315-318
    },
    'RoF': {
        '0007': 282,
        '0014': 286,
        '0021': 290,
    }

}


class CwruDataset(Dataset):
    """返回一维时序数据的原始数据集"""

    def __init__(self, root, load_hp: Union[list, int] = None, load_bearings: Union[list, str] = None,
                 n_samples=100, n_points=1024, overlap=100, normalize='0-1'):
        """
        故障类别： IF, OF, RoF, Normal, 故障直径： 0.007, 0.014, 0.021， 因此共计3*3+1=10个类别
        考虑四种工况条件，最多10*4=40个类别
        考虑驱动端和风扇端，共计40*2=80个类别
        :param root: 根路径
        :param load_hp: 工作负载，可选 [0, 1, 2, 3]
        :param load_bearings: 轴承类型，可选 ["DE", "FE"]
        :param n_samples: 样本数
        :param n_points: 每个样本包含的数据点数
        :param normalize:
        """
        # normal_path CWRU正常数据的文件夹路径
        # de_fault_12k_path CWRU中12K采频数据的文件夹路径

        self.n_points = n_points
        self.n_samples = n_samples
        self.overlap = overlap
        self.normal_path = os.path.join(root, 'Normal Baseline Data')
        self.de_fault_12k_path = os.path.join(root, '12k Drive End Bearing Fault Data')
        self.fe_fault_12k_path = os.path.join(root, 'Fan-End Bearing Fault Data')
        self.fault_diameter = [0.007, 0.014, 0.021]
        self.load_hp = load_hp if load_hp is not None else [0, 1, 2, 3]
        if load_bearings is None:
            load_bearings = ['DE', 'FE']
        elif not isinstance(load_bearings, list) or not isinstance(load_bearings, tuple):
            load_bearings = [load_bearings.upper()]
        else:
            load_bearings = [load_bearing.upper() for load_bearing in load_bearings]
        self.load_bearings = load_bearings
        self.need_scale = normalize

        self.transform = transforms.Compose([
            aug.Normalize(normalize),
            # aug.Reshape(),
        ])

        self.x, self.y, self.hp, self.id2label = self.load_data()
        self.num_classes = len(self.id2label.values())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x, y, hp = self.x[item], self.y[item], self.hp[item]
        seq = self.transform(torch.from_numpy(x)).reshape(-1, 1)  # [n_points_per_sample, 1]

        return seq, y

    def get_label_name(self, y):
        y = int(y)
        return self.id2label[y]

    def load_data(self):
        x, y, hp = [], [], []
        labels, len_count = [], []
        id2label = {}

        # read drive end bearing data
        if "DE" in self.load_bearings:
            for i, p in enumerate(self.load_hp):
                # 正常数据
                normal_de = open_data(self.normal_path, 97 + p, name='_DE_time')
                normal_de_preprocessed = preprocess_data(normal_de, self.n_points, self.n_samples, self.overlap)
                x.append(normal_de_preprocessed)
                labels.append(f'Normal-{p}-DE')
                len_count.append(len(normal_de_preprocessed))

                # 故障数据
                for j, diameter in enumerate(self.fault_diameter):
                    if diameter == 0.007:
                        inner_num = 105
                        ball_num = 118
                        outer_num = 130
                    elif diameter == 0.014:
                        inner_num = 169
                        ball_num = 185
                        outer_num = 197
                    else:
                        inner_num = 209
                        ball_num = 222
                        outer_num = 234

                    inner_de = open_data(self.de_fault_12k_path, inner_num + p)
                    inner_de_preprocessed = preprocess_data(inner_de, self.n_points, self.n_samples, self.overlap)
                    x.append(inner_de_preprocessed)
                    labels.append(f'IF-{p}-DE')
                    len_count.append(len(inner_de_preprocessed))

                    ball_de = open_data(self.de_fault_12k_path, ball_num + p)
                    ball_de_preprocessed = preprocess_data(ball_de, self.n_points, self.n_samples, self.overlap)
                    x.append(ball_de_preprocessed)
                    labels.append(f'RoF-{p}-DE')
                    len_count.append(len(ball_de_preprocessed))

                    outer_de = open_data(self.de_fault_12k_path, outer_num + p)
                    outer_de_preprocessed = preprocess_data(outer_de, self.n_points, self.n_samples, self.overlap)
                    x.append(outer_de_preprocessed)
                    labels.append(f'OF-{p}-DE')
                    len_count.append(len(outer_de_preprocessed))

        # read fan end bearing data
        if "FE" in self.load_bearings:
            for i, p in enumerate(self.load_hp):
                # 正常数据
                normal_fe = open_data(self.normal_path, 97 + p, name='_FE_time')
                normal_fe_preprocessed = preprocess_data(normal_fe, self.n_points, self.n_samples, self.overlap)
                x.append(normal_fe_preprocessed)
                labels.append(f'Normal-{p}-FE')
                len_count.append(len(normal_fe_preprocessed))

                for j, diameter in enumerate(self.fault_diameter):
                    if diameter == 0.007:
                        inner_num = FE_DATA_DICT['IF']['0007']
                        ball_num = FE_DATA_DICT['RoF']['0007']
                        outer_num = FE_DATA_DICT['OF']['0007']
                    elif diameter == 0.014:
                        inner_num = FE_DATA_DICT['IF']['0014']
                        ball_num = FE_DATA_DICT['RoF']['0014']
                        outer_num = FE_DATA_DICT['OF']['0014']
                    else:
                        inner_num = FE_DATA_DICT['IF']['0021']
                        ball_num = FE_DATA_DICT['RoF']['0021']
                        outer_num = FE_DATA_DICT['OF']['0021']

                    inner_fe = open_data(self.fe_fault_12k_path, inner_num + p, name='_FE_time')
                    inner_fe_preprocessed = preprocess_data(inner_fe, self.n_points, self.n_samples, self.overlap)
                    x.append(inner_fe_preprocessed)
                    labels.append(f'IF-{p}-FE')
                    len_count.append(len(inner_fe_preprocessed))

                    ball_fe = open_data(self.fe_fault_12k_path, ball_num + p, name='_FE_time')
                    ball_fe_preprocessed = preprocess_data(ball_fe, self.n_points, self.n_samples, self.overlap)
                    x.append(ball_fe_preprocessed)
                    labels.append(f'RoF-{p}-FE')
                    len_count.append(len(ball_fe_preprocessed))

                    outer_fe = open_data(self.fe_fault_12k_path, outer_num + p, name='_FE_time')
                    outer_fe_preprocessed = preprocess_data(outer_fe, self.n_points, self.n_samples, self.overlap)
                    x.append(outer_fe_preprocessed)
                    labels.append(f'OF-{p}-FE')
                    len_count.append(len(outer_fe_preprocessed))

        x = np.concatenate(x)
        # if self.need_scale:
        #     scaler = preprocessing.MinMaxScaler()
        #     x = scaler.fit_transform(np.transpose(x, [1, 0]))
        #     x = np.transpose(x, [1, 0])
        # 重新分配标签并记录标签索引与标签名的关系
        for i, (label, length) in enumerate(zip(labels, len_count)):
            y += [i] * length
            hp += [int(label.split('-')[1])] * length
            id2label[i] = label
        # x size: [n_samples_per_class*n_load_hp*10, n_points_per_sample]
        # y size: [n_samples_per_class*n_load_hp*10, ]
        # hp size: [n_samples_per_class*n_load_hp*10, ]

        return x, y, hp, id2label


class CwruMultiModalDataset(CwruDataset):
    """返回多模态的数据集"""

    def __init__(self, root, freq_trans_fn, img_trans_fn, load_hp: Union[list, int] = None,
                 load_bearings: Union[list, str] = None,
                 n_samples=120, n_points=1024, overlap=100, normalize='0-1', img_size=256):
        super().__init__(root, load_hp, load_bearings, n_samples, n_points, overlap, normalize)
        self.img_trans_fn = img_trans_fn
        self.freq_trans_fn = freq_trans_fn

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        x, y, hp = self.x[item], self.y[item], self.hp[item]
        seq = self.transform(torch.from_numpy(x)).reshape(-1, 1)  # [n_points_per_sample, 1]

        # freq = self.freq_trans_fn(np.reshape(x, (1, -1)))
        freq = self.freq_trans_fn(x)
        # freq = self.transform(torch.from_numpy(freq).squeeze(0))
        freq = self.transform(torch.from_numpy(freq)).reshape(-1, 1)

        img = self.img_trans_fn(np.reshape(x, (1, -1)))
        img = self.img_transform(img)

        # seq = torch.from_numpy(x).unsqueeze(1)  # [n_points_per_sample, 1]

        return seq, freq, img, y


class CwruTFDataset(CwruDataset):
    """only time and frequency modalities"""
    def __init__(self, root, freq_trans_fn, load_hp: Union[list, int] = None,
                 load_bearings: Union[list, str] = None,
                 n_samples=120, n_points=1024, overlap=100, normalize='0-1'):
        super().__init__(root, load_hp, load_bearings, n_samples, n_points, overlap, normalize)
        self.freq_trans_fn = freq_trans_fn

    def __getitem__(self, item):
        x, y, hp = self.x[item], self.y[item], self.hp[item]
        seq = self.transform(torch.from_numpy(x)).reshape(-1, 1)  # [n_points_per_sample, 1]

        freq = self.freq_trans_fn(x)    # size (n_points_per_sample)
        freq = self.transform(torch.from_numpy(freq)).reshape(-1, 1)

        return seq, freq, y


class CwruFreqDataset(CwruDataset):
    def __init__(self, root, freq_trans_fn, load_hp: Union[list, int] = None, load_bearings: Union[list, str] = None,
                 n_samples=120, n_points=1024, overlap=100, normalize='0-1'):
        super().__init__(root, load_hp, load_bearings, n_samples, n_points, overlap, normalize)
        self.freq_trans_fn = freq_trans_fn

    def __getitem__(self, item):
        x, y, hp = self.x[item], self.y[item], self.hp[item]

        freq = self.freq_trans_fn(x)  # size (n_points_per_sample)
        freq = self.transform(torch.from_numpy(freq)).reshape(-1, 1)

        return freq, y


class CwruVisualDataset(CwruDataset):
    """CWRU视觉域数据集"""

    def __init__(self, root, img_trans_fn, load_hp: Union[list, int] = None, load_bearings: Union[list, str] = None,
                 n_samples=120, n_points=1024, overlap=100, normalize='0-1', img_size=256):
        super().__init__(root, load_hp, load_bearings, n_samples, n_points, overlap, normalize)
        self.img_trans_fn = img_trans_fn
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        x, y, hp = self.x[item], self.y[item], self.hp[item]
        img = self.img_trans_fn(np.reshape(x, (1, -1)))
        img = self.img_transform(img)

        return img, y