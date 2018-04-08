import collections

import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, input_dim):
        super(Dense, self).__init__()

        self.layers = nn.Sequential(collections.OrderedDict([
            ('linear-0', nn.Linear(input_dim, input_dim)),
            ('relu', nn.ReLU()),
            ('linear-1', nn.Linear(input_dim, input_dim))
        ]))

    def forward(self, sequence):
        return sequence + self.layers(sequence)


class ConvBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 filter_mapping,
                 dropout_prob):

        super(ConvBlock, self).__init__()

        self.convs = nn.ModuleList()
        for filter_size, num_filters in filter_mapping.items():
            if filter_size % 2 == 0:
                raise ValueError('filter size cannot be even')

            self.convs.append(nn.Conv2d(1,
                                        num_filters,
                                        (filter_size, input_dim),
                                        padding=((filter_size - 1) // 2, 0)))

        self.dense = Dense(sum(filter_mapping.values()))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sequence):
        sequence = sequence.unsqueeze(1)

        feature_maps = [conv(sequence).squeeze(-1) for conv in self.convs]
        feature_maps = torch.cat(feature_maps, 1).transpose(1, 2)

        return self.dropout(self.relu(self.dense(feature_maps)))
