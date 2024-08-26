# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      mmfd
   Author :         zmfy
   DateTime :       2024/5/18 16:42
   Description :    
-------------------------------------------------
"""

from networks.proto import MetricModelBase
from networks.fusion import *
from networks.encoders import Conv64F, TemporalConvNet
from utils import *


class MultiModalFD(MetricModelBase):
    def __init__(self, config, input_shape, device=None):
        super().__init__(config)
        ts_shape, freq_shape, img_shape = input_shape
        self.img_size = img_shape
        self.freq_shape = freq_shape
        self.ts_shape = ts_shape
        self.config = config
        # e.g., ts_shape = (n_points_per_sample, 1), freq_shape = (n_dim, 1), img_size = 224

        self.ts_encoder = TemporalConvNet(in_planes=ts_shape[-1], out_planes=config.ts_tcn_channels,
                                          num_layers=config.ts_tcn_layers, kernel_size=config.ts_tcn_ksize)
        self.freq_encoder = TemporalConvNet(in_planes=freq_shape[-1], out_planes=config.freq_tcn_channels,
                                            num_layers=config.freq_tcn_layers, kernel_size=config.freq_tcn_ksize)
        self.img_encoder = Conv64F(in_planes=img_shape[0], out_channels=64)
        x = torch.randn((1, *img_shape))
        with torch.no_grad():
            out = self.img_encoder(x).reshape(1, -1)
        img_out_features = out.size(1)
        x = torch.randn((1, *ts_shape))
        with torch.no_grad():
            out = self.ts_encoder(x).reshape(1, -1)
        ts_out_features = out.size(1)

        # dimension reshaper
        out_features = config.hidden_size
        self.ts_proj = nn.Linear(in_features=ts_out_features, out_features=out_features)
        self.freq_proj = nn.Linear(in_features=ts_out_features, out_features=out_features)
        self.img_proj = nn.Linear(in_features=img_out_features, out_features=out_features)

        # fusion network
        self.fusion_net = FusionMLP(in_dim=out_features, n_hiddens=config.mlp_hidden)

    def init_weights(self, init_type, init_gain=0.02):
        init_weights(self.ts_encoder, init_type, init_gain=init_gain)
        init_weights(self.freq_encoder, init_type, init_gain=init_gain)
        init_weights(self.ts_proj, init_type, init_gain=init_gain)
        init_weights(self.freq_proj, init_type, init_gain=init_gain)
        init_weights(self.img_proj, init_type, init_gain=init_gain)
        init_weights(self.img_encoder, init_type, init_gain=init_gain)
        init_weights(self.fusion_net, init_type, init_gain=init_gain)
        logging.info(f'initialize network with {init_type}')

    def forward(self, x, labels, k_shot, **kwargs):
        """

        :param x: tuple of (ts, freq, img)
        :param k_shot: int
        :param labels: size of (n_way*n_query, )
        :return:
        """
        ts, freq, img = x
        assert ts.size(0) == freq.size(0) == img.size(0)

        embedding = self.encode(ts, freq, img)
        return self.prototypical_head(embedding, labels, k_shot)

    def encode(self, ts, freq, img):
        # embedding
        ts_embedding = self.ts_encoder(ts).reshape(ts.size(0), -1)  # size: [bsize, dim]
        freq_embedding = self.freq_encoder(freq).reshape(ts.size(0), -1)  # size: [bsize, dim]
        img_embedding = self.img_encoder(img).reshape(ts.size(0), -1)  # size: [bsize, dim_image]
        ts_embedding = self.ts_proj(ts_embedding)
        freq_embedding = self.freq_proj(freq_embedding)
        img_embedding = self.img_proj(img_embedding)

        # fusion
        h_1 = self.fusion_net(freq_embedding)  # (bsize, 1)
        h_2 = self.fusion_net(img_embedding)  # (bsize, 1)
        lam_1 = 1. / (1 + torch.exp(-h_1))
        lam_2 = 1. / (1 + torch.exp(-h_2))

        embedding = lam_1 * ts_embedding + (1 - lam_1) * freq_embedding
        embedding = lam_2 * embedding + (1 - lam_2) * img_embedding

        return embedding


class UniModalFD(MetricModelBase):
    def __init__(self, config, input_shape, modal='time'):
        super(UniModalFD, self).__init__(config)
        self.input_shape = input_shape
        self.modal = modal

        if modal == 'time':
            num_outputs = self.config.ts_tcn_channels
            self.encoder = TemporalConvNet(in_planes=input_shape[-1], out_planes=num_outputs,
                                           num_layers=config.ts_tcn_layers, kernel_size=config.ts_tcn_ksize)
        elif modal == 'freq':
            num_outputs = self.config.freq_tcn_channels
            self.encoder = TemporalConvNet(in_planes=input_shape[-1], out_planes=num_outputs,
                                           num_layers=config.freq_tcn_layers, kernel_size=config.freq_tcn_ksize)
        elif modal == 'visual':
            self.encoder = Conv64F(in_planes=input_shape[0], out_channels=64)
        else:
            raise ValueError(f'param "modal" must be time, freq or image, not {self.modal}')

    def init_weights(self, init_type, init_gain=0.02):
        init_weights(self.encoder, init_type, init_gain=init_gain)
        logging.info(f'initialize network with {init_type}')

    def forward(self, x, labels, k_shot, **kwargs):
        embedding = self.encoder(x).reshape(x.size(0), -1)

        return self.prototypical_head(embedding, labels, k_shot)