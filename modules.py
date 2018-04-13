import collections

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 filter_mapping):

        super(ConvLayer, self).__init__()

        self.convs = nn.ModuleList()
        for filter_size, num_filters in filter_mapping.items():
            if filter_size % 2 == 0:
                raise ValueError('filter size cannot be even')

            self.convs.append(nn.Conv2d(1,
                                        num_filters,
                                        (filter_size, input_dim),
                                        padding=((filter_size - 1) // 2, 0)))

    def forward(self, sequence):
        sequence = sequence.unsqueeze(1)

        feature_maps = [conv(sequence).squeeze(-1) for conv in self.convs]
        feature_maps = torch.cat(feature_maps, 1).transpose(1, 2)

        return feature_maps


class ConvBlock(nn.Module):
    def __init__(self,
                 filter_mapping):

        super(ConvBlock, self).__init__()

        self.input_dim = sum(filter_mapping.values())

        self.layers = nn.Sequential(collections.OrderedDict([
            ('conv-0', ConvLayer(self.input_dim, filter_mapping)),
            ('relu-0', nn.ReLU()),
            ('conv-1', ConvLayer(self.input_dim, filter_mapping))
        ]))

        self.relu = nn.ReLU()

    def forward(self, sequence):
        return self.relu(sequence + self.layers(sequence))
