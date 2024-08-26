# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      config
   Author :         zmfy
   DateTime :       2024/5/20 12:13
   Description :    
-------------------------------------------------
"""
import datetime
import json
import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            self.base_path = r'/opt/data/private/Meta-Learning/FaultDiagnosis/MMPN-FD'
            self.seed = 520  # -1 represents no fixed seed
            self._init_model_config()
            self._init_train_config()

    def _init_model_config(self):
        # 以下是模型参数
        self.modal = 'all'  # 'all', 'time', 'freq', 'visual'
        self.ts_tcn_layers = 2  # TCN 层数
        self.ts_tcn_channels = [8, 16]  # TCN 的输出通道数
        self.ts_tcn_ksize = [64, 3]  # TCN 每个一维卷积的kernel_size， 一维卷积数 = 2 * TCN 层数
        assert len(self.ts_tcn_channels) == self.ts_tcn_layers
        self.freq_tcn_layers = 2  # TCN 层数
        self.freq_tcn_channels = [8, 16]  # TCN 的输出通道数
        self.freq_tcn_ksize = [64, 3]
        self.hidden_size = 1024  # 特征融合前的维度
        self.mlp_hidden = 128
        self.use_center_loss = True
        self.center_loss_multiplier = 0.3  # Center Loss的权值

    def _init_train_config(self):
        self.use_gpu = True
        self.pretrain = False  # 加载预训练模型时设置为 True

        # 输入数据集、日志、输出目录
        self.data_name = 'CWRU'
        self.task_type = 0  # 0 for cross-workload, 1 for cross-bearing
        self.task_name = 'CL3'
        assert ((self.task_type == 0 and self.task_name[:2] == "CL") or
                (self.task_type == 1 and self.task_name[:2] == "CB"))

        self.data_dir = os.path.join('/opt/data/private/dataset/', self.data_name)
        self.n_samples_per_class = 100
        self.n_points_per_sample = 1024
        self.shift_size = 100  # 滑动窗口shift
        self.normalize = "-1-1"
        self.img_size = 84
        self.image_transform_method = 'GASF'
        self.log_name = 'train'

        # 子数据集定义
        self.datasets = {
            'CL1': {"train_hp": (0, 1, 2), 'eval_hp': 3},
            'CL2': {"train_hp": (0, 1, 3), 'eval_hp': 2},
            'CL3': {"train_hp": (0, 2, 3), 'eval_hp': 1},
            'CL4': {"train_hp": (1, 2, 3), 'eval_hp': 0},
            'CB1': {'train_bearings': "DE", 'eval_bearings': "FE"},
            'CB2': {'train_bearings': "DE", 'eval_bearings': "FE"},
        }

        # 模型保存路径
        now = datetime.datetime.now()
        t = now.strftime('%m-%d-%H-%M')
        self.trained_model_dir = os.path.join(self.base_path, 'output', self.data_name, t)

        # 以下是模型训练参数
        self.do_eval = True  # 每训练一个epoch 验证一次
        self.do_test = False  # 训练完测试
        self.n_way = 5
        self.n_way_tsne = 10
        self.k_shot_tr = 1
        self.k_shot_eval = 1
        self.n_query_tr = 10
        self.n_query_eval = 10
        self.iter_tr = 100
        self.iter_eval = 50
        self.iter_test = 1000
        self.num_train_epochs = 10
        self.initial_lr = 5e-4
        self.min_lr = 1e-6
        self.scheduler = "cosine"  # cosine-warm or cosine
        self.logging_steps = 100
        self.init_type = 'xavier'
        self.init_gain = 0.02
        self.loader_workers = 10

        # 输入形状自动计算
        self.ts_shape = (self.n_points_per_sample, 1)
        self.freq_shape = self.ts_shape
        self.img_shape = (3, self.img_size, self.img_size)

        if self.modal == 'all':
            self.input_shape = (self.ts_shape, self.ts_shape, self.img_shape)
        elif self.modal == 'time':
            self.input_shape = self.ts_shape
        elif self.modal == 'freq':
            self.input_shape = self.freq_shape
        elif self.modal == 'visual':
            self.input_shape = self.img_shape
        else:
            raise ValueError

        sub_dir = f'{self.task_name}'

    @classmethod
    def init_from_json(cls, json_path):
        # 从json初始化Config
        with open(json_path, 'r') as f:
            config_dic = json.load(f)

        for key, value in config_dic.items():
            setattr(cls, key, value)

        return cls

