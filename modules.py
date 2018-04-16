import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 input_channels,
                 num_filters,
                 filter_size,
                 num_groups):

        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(input_channels,
                              num_filters,
                              (filter_size, input_dim),
                              padding=((filter_size - 1) // 2, 0),
                              groups=num_groups)

        self.norm = nn.BatchNorm2d(num_filters)

    def forward(self, sequence):
        return self.norm(self.conv(sequence))


class ConvBlock(nn.Module):
    def __init__(self,
                 num_filters,
                 filter_size,
                 num_groups):

        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential()

        self.layers.add_module('conv-0', ConvLayer(1,
                                                   num_filters,
                                                   num_filters,
                                                   filter_size,
                                                   num_groups))

        self.layers.add_module('relu-0', nn.ReLU())

        self.layers.add_module('conv-1', ConvLayer(1,
                                                   num_filters,
                                                   num_filters,
                                                   filter_size,
                                                   num_groups))

        self.relu = nn.ReLU()

    def forward(self, sequence):
        return self.relu(sequence + self.layers(sequence))
