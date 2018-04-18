import torch.nn as nn


def get_padding(filter_size):
    return ((filter_size - 1) // 2, 0)


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
                              padding=get_padding(filter_size),
                              groups=num_groups)

    def forward(self, sequence):
        return self.conv(sequence)


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


class MaxPoolLayer(nn.Module):
    def __init__(self,
                 num_filters,
                 filter_size):

        super(MaxPoolLayer, self).__init__()

        self.pool = nn.MaxPool2d((filter_size, 1),
                                 padding=get_padding(filter_size))

    def forward(self, sequence):
        return self.pool(sequence)
