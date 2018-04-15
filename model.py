import torch
import torch.nn as nn

from modules import ConvLayer, ConvBlock


class CNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 labelset_size,
                 embedding_dim,
                 num_layers,
                 num_filters,
                 filter_size,
                 num_groups,
                 dropout_prob=.5,
                 pretrained_embeddings=None):

        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.layers = nn.Sequential()
        self.layers.add_module('conv-0',
                               ConvLayer(embedding_dim,
                                         1,
                                         num_filters,
                                         filter_size,
                                         num_groups=1))

        self.layers.add_module('relu-0', nn.ReLU())

        for i in range(num_layers):
            self.layers.add_module(f'conv-block-{i}',
                                   ConvBlock(num_filters,
                                             filter_size,
                                             num_groups))

        self.linear = nn.Linear(num_filters, labelset_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        embeddings = self.embedding(sequence)
        embeddings = embeddings.unsqueeze(1)

        feature_maps = self.layers(embeddings)
        feature_maps = feature_maps.squeeze(-1)

        feature_vec, _ = torch.max(feature_maps, 2)
        feature_vec = self.dropout(feature_vec)

        return self.softmax(self.linear(feature_vec))
