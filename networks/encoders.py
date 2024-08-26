# -*- coding:utf-8 -*-
# @FileName  :encoders.py
# @Time      :2024/7/10 下午12:44
# @Author    :ZMFY
# Description:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def conv1d_block(inplanes, planes, kernel_size, stride=1, padding=0, with_pool=True):
    return nn.Sequential(
        nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2) if with_pool else nn.Identity()
    )


def conv_block(in_channel, out_channel, kernel_size=3, pool_size=2, with_pooling=True):
    """Return a block of conv2d-bn-relu-maxpooling"""
    nets = [
        nn.Conv2d(in_channel, out_channel, kernel_size, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.PReLU(),
    ]
    if with_pooling:
        nets += [nn.MaxPool2d(pool_size)]
    return nn.Sequential(*nets)


class Conv64F(nn.Module):
    """Define the embedding network"""

    def __init__(self, in_planes, out_channels=64, num_hiddens=64, num_layers=4) -> None:
        super().__init__()
        assert num_layers > 2

        self.embedding = []
        self.embedding += [conv_block(in_planes, num_hiddens, kernel_size=3, pool_size=2)]
        for i in range(num_layers - 2):
            self.embedding += [conv_block(num_hiddens, num_hiddens)]
        self.embedding += [conv_block(num_hiddens, out_channels)]
        self.embedding = nn.Sequential(*self.embedding)

    def forward(self, x):
        return self.embedding(x)

    def __iter__(self):
        return iter(self.embedding)


class Chomp1d(nn.Module):
    def __init__(self, size, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size = size

    def forward(self, X):
        return X[:, :, :-self.size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, dilation,
                 kernel_sizes=[2, 2], dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        paddings = [(kernel_size - 1) * dilation for kernel_size in kernel_sizes]
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(
                num_inputs, num_outputs, kernel_sizes[0], padding=paddings[0],
                stride=1, dilation=dilation,
            )),
            nn.BatchNorm1d(num_outputs),
            Chomp1d(paddings[0]),
            nn.PReLU(),
            nn.Dropout(dropout),

            weight_norm(nn.Conv1d(
                num_outputs, num_outputs, kernel_sizes[1], padding=paddings[1],
                stride=1, dilation=dilation,
            )),
            nn.BatchNorm1d(num_outputs),
            Chomp1d(paddings[1]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.downsampling = nn.Conv1d(num_inputs, num_outputs, 1) if num_inputs != num_outputs else None
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, X):
        Y = self.net(X)
        if self.downsampling:
            X = self.downsampling(X)
        return self.pool(self.relu(X + Y))

    def __iter__(self):
        layers = [layer for layer in self.net]
        return iter(layers)


class TemporalConvNet(nn.Module):
    def __init__(self, in_planes, out_planes: [list, int],
                 num_layers=2, kernel_size: [list, int] = 2, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        layers = []
        if isinstance(out_planes, list):
            assert len(out_planes) == num_layers
        elif isinstance(out_planes, int):
            out_planes = [out_planes] * num_layers
        else:
            raise TypeError("The param of 'out_planes' must be 'list' or 'int' method.")
        if isinstance(kernel_size, int):
            kernel_size = [[kernel_size] * 2] * num_layers
        elif isinstance(kernel_size, list) and isinstance(kernel_size[0], int):
            kernel_size = [[kernel_size[i]] * 2 for i in range(num_layers)]
        for i in range(num_layers):
            num_output = out_planes[i]
            ksizes = kernel_size[i]
            dilation_size = 2 ** i
            layers.append(
                TemporalBlock(
                    in_planes, num_output, kernel_sizes=ksizes,
                    dilation=dilation_size, dropout=dropout
                )
            )
            in_planes = num_output
        self.featrures = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x shape : (batch_size, num_features, time_steps or num_steps)
        x = self.featrures(x).permute(0, 2, 1)

        return x

    def __iter__(self):
        layers = []
        for block in self.featrures:
            layers.extend([layer for layer in block])
        # layers.extend([layer for layer in self.fc])
        return iter(layers)
