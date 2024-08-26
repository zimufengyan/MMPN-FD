# -*- coding:utf-8 -*-
# @FileName  :proto.py
# @Time      :2024/7/4 下午3:34
# @Author    :ZMFY
# Description:

from networks.encoders import *
from utils import *
from networks.fusion import CenterLoss


class MetricModelBase(nn.Module):
    def __init__(self, config):
        super(MetricModelBase, self).__init__()
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        if config.use_center_loss:
            self.center_criterion = CenterLoss(p=2, reduction='mean')

    def init_weights(self, init_type, init_gain=0.02):
        raise NotImplementedError

    def forward(self, x, labels, k_shot, **kwargs):
        raise NotImplementedError

    def prototypical_head(self, embedding, labels, k_shot):
        # 原型网络分类头, labels only for query set
        n_way = self.config.n_way
        mult = self.config.center_loss_multiplier

        # emd_s size (n_way*k_shot, ts_shape[0]), emd_q size (n_way*k_shot, ts_shape[0])
        emd_s, emd_q = embedding[:n_way * k_shot], embedding[n_way * k_shot:]

        # computing prototypical centers
        # emd_s size (n_way, k_*shot, ts_shape[0])
        emd_s = emd_s.reshape(k_shot, n_way, emd_s.size(1)).permute(1, 0, 2).contiguous()
        prototypes = emd_s.mean(dim=1)

        # computing similarity and loss
        similarity = compute_similarity(emd_q, prototypes, method='euclidean')
        loss = self.criterion(similarity, labels)

        if self.config.use_center_loss:
            c_loss = self.center_criterion(emd_s, prototypes)
            loss = loss + mult * c_loss

        return similarity, loss