# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      main
   Author :         zmfy
   DateTime :       2024/5/20 12:13
   Description :    
-------------------------------------------------
"""
import logging
import os
import sys

import torch

from config import Config
from trainer import MMFDTrainer
from utils import *


def main():
    config = Config()
    modal = config.modal

    set_seed(config.seed)

    logger_dir = config.trained_model_dir
    set_logger(logger_dir, config.log_name)
    logging.info(f"current modal: {modal}")
    logging.info(f'task_name : {config.task_name}')

    use_gpu = torch.cuda.is_available() and config.use_gpu
    config.use_gpu = use_gpu
    config.device = 'cuda' if use_gpu else 'cpu'
    device = torch.device('cuda' if use_gpu else 'cpu')
    n_gpu = torch.cuda.device_count()
    logging.info(f"available device: {device}，count_gpu: {n_gpu}")

    input_shape = config.input_shape
    logging.info(str(input_shape))

    trainer = MMFDTrainer(config, input_shape=input_shape, modal=modal)
    trainer.train()
    if config.do_test:
        trainer.test()


if __name__ == '__main__':
    main()
