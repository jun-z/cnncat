import torch
import torch.nn as nn

from modules import ConvLayer, ConvBlock


class CNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 labelset_size,
                 embedding_dim,
                 num_layers,
                 filter_mapping,
                 dropout_prob=.5,
                 pretrained_embeddings=None):

        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.convs = nn.Sequential()
        self.convs.add_module('conv-layer-0',
                              ConvLayer(embedding_dim, filter_mapping))

        for i in range(num_layers):
            self.convs.add_module(f'conv-block-{i}',
                                  ConvBlock(filter_mapping))

        self.linear = nn.Linear(sum(filter_mapping.values()), labelset_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        embeddings = self.embedding(sequence)

        feature_maps = self.convs(embeddings)

        feature_vec, _ = torch.max(feature_maps, 1)
        feature_vec = self.dropout(feature_vec)

        return self.softmax(self.linear(feature_vec))
