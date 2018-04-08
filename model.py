import torch
import torch.nn as nn

from modules import ConvBlock


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
        for i in range(num_layers):
            if i == 0:
                self.convs.add_module(f'conv-{i}',
                                      ConvBlock(embedding_dim,
                                                filter_mapping,
                                                dropout_prob))
            else:
                self.convs.add_module(f'conv-{i}',
                                      ConvBlock(sum(filter_mapping.values()),
                                                filter_mapping,
                                                dropout_prob))

        self.linear = nn.Linear(sum(filter_mapping.values()), labelset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        embeddings = self.embedding(sequence)

        feature_maps = self.convs(embeddings)
        feature_vec, _ = torch.max(feature_maps, 1)

        return self.softmax(self.linear(feature_vec))
