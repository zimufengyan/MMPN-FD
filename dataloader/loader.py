# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      loader
   Author :         zmfy
   DateTime :       2024/5/18 19:53
   Description :    
-------------------------------------------------
"""
import logging

from scipy.fftpack import fft
from torch.utils.data import DataLoader

from dataloader.dataset import (CwruDataset, CwruMultiModalDataset,
                                CwruVisualDataset, CwruFreqDataset)
from dataloader.sampler import FewShotBatchSampler
from dataloader.tools import ts2img_factor


def init_dataloader(config, mode='train'):
    assert config.task_type == 0 or config.task_type == 1
    n_way = config.n_way
    if mode == 'train':
        load_hp = config.datasets[config.task_name]['train_hp'] if config.task_type == 0 else None
        load_bearings = config.datasets[config.task_name]['train_bearings'] if config.task_type == 1 else None
        n_query_tr = config.n_query_tr
        k_shot = config.k_shot_tr + n_query_tr
        iterations = config.iter_tr
    else:
        load_hp = config.datasets[config.task_name]['eval_hp'] if config.task_type == 0 else None
        load_bearings = config.datasets[config.task_name]['eval_bearings'] if config.task_type == 1 else None
        if isinstance(load_hp, int):
            load_hp = [load_hp]
        k_shot = config.k_shot_eval + config.n_query_eval
        iterations = config.iter_eval if mode == 'eval' else config.iter_test

    fft_trans = lambda x: abs(fft(x))
    img_trans_fn = lambda x: ts2img_factor(x, img_size=config.n_points_per_sample,
                                           trans_method=config.image_transform_method,
                                           need_scale=False if config.normalize else True)
    if config.modal == 'time':
        dataset = CwruDataset(config.data_dir, load_hp=load_hp, load_bearings=load_bearings,
                              n_samples=config.n_samples_per_class, n_points=config.n_points_per_sample,
                              normalize=config.normalize)
    elif config.modal == 'visual':
        dataset = CwruVisualDataset(config.data_dir, img_trans_fn=img_trans_fn,
                                    load_hp=load_hp, load_bearings=load_bearings,
                                    n_samples=config.n_samples_per_class, n_points=config.n_points_per_sample,
                                    normalize=config.normalize)
    elif config.modal == 'freq':
        dataset = CwruFreqDataset(config.data_dir, freq_trans_fn=fft_trans,
                                  load_hp=load_hp, load_bearings=load_bearings,
                                  n_samples=config.n_samples_per_class, n_points=config.n_points_per_sample,
                                  normalize=config.normalize)
    else:
        dataset = CwruMultiModalDataset(
            config.data_dir, freq_trans_fn=fft_trans, img_trans_fn=img_trans_fn, load_hp=load_hp,
            load_bearings=load_bearings, n_samples=config.n_samples_per_class, n_points=config.n_points_per_sample,
            normalize=config.normalize, img_size=config.img_size
        )

    sampler = FewShotBatchSampler(dataset.y, n_ways=n_way, k_shots=k_shot, iterations=iterations)
    loader = DataLoader(
        dataset, batch_sampler=sampler, pin_memory=True if config.use_gpu else False,
        num_workers=config.loader_workers
    )
    logging.info(f"read {mode} dataset successfully. number of classes: {dataset.num_classes}")
    return loader
