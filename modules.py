import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self,
                 num_input_filters,
                 growth_rate,
                 filter_size,
                 padding,
                 dropout_prob):

        super(DenseLayer, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_filters))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(num_input_filters,
                                          growth_rate,
                                          (1, filter_size),
                                          padding=padding))

        self.dropout = nn.Dropout()

    def forward(self, feature_maps):
        new_feature_maps = super(DenseLayer, self).forward(feature_maps)
        new_feature_maps = self.dropout(new_feature_maps)
        return torch.cat([feature_maps, new_feature_maps], 1)


class DenseBlock(nn.Sequential):
    def __init__(self,
                 num_input_filters,
                 num_layers,
                 growth_rate,
                 filter_size,
                 dropout_prob):

        super(DenseBlock, self).__init__()

        if filter_size % 2 == 0:
            raise ValueError('filter size cannot be even')

        for i in range(num_layers):
            self.add_module(f'dense-{i}',
                            DenseLayer(num_input_filters + i * growth_rate,
                                       growth_rate,
                                       filter_size,
                                       padding=(0, (filter_size - 1) // 2),
                                       dropout_prob=dropout_prob))

        self.add_module('activation', nn.ReLU())
