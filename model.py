import torch
import torch.nn as nn

from modules import Dense, ConvBlock


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

        self.hidden_dim = sum(filter_mapping.values())

        self.convs = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.convs.add_module(f'conv-{i}',
                                      ConvBlock(embedding_dim,
                                                filter_mapping,
                                                dropout_prob))
            else:
                self.convs.add_module(f'conv-{i}',
                                      ConvBlock(self.hidden_dim,
                                                filter_mapping,
                                                dropout_prob))

        self.dense = Dense(self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, labelset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        embeddings = self.embedding(sequence)

        feature_maps = self.convs(embeddings)
        feature_vec, _ = torch.max(feature_maps, 1)
        feature_vec = self.dense(feature_vec)

        return self.softmax(self.linear(feature_vec))
