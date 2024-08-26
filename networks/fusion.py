# -*- coding:utf-8 -*-
# @FileName  :base_net.py
# @Time      :2024/6/1 下午8:14
# @Author    :ZMFY
# Description:

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, in_dim, n_hiddens=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, n_hiddens),
            nn.BatchNorm1d(n_hiddens),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hiddens, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class CenterLoss(nn.Module):
    """
    实现中心损失

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, p=2, reduction='sum'):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, embeddings, centers):
        """

        :param embeddings: embeddings of samples from multi classes, size of (n_way, k_shot, feature_dim),
                            label form: aabbccdd
        :param centers: centers of multi classes, size of (n_way, feature_dim), label form: abcd
        :return:
        """
        return center_loss(embeddings, centers, self.p, self.reduction)


def center_loss(embeddings, centers, p=2, reduction='mean'):
    n_way = centers.size(0)
    k_shot = embeddings.size(1)
    centers = centers.unsqueeze(1).expand(n_way, k_shot, centers.size(1))
    losses = torch.norm(embeddings - centers, p=p, dim=2, keepdim=False).sum(dim=1)
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        return losses


if __name__ == "__main__":
    pass
