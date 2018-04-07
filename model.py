import torch.nn as nn

from modules import DenseBlock


class CNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 labelset_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 growth_rate,
                 filter_size,
                 dropout_prob=.5,
                 pretrained_embeddings=None):

        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.dense_block = DenseBlock(embedding_dim,
                                      num_layers,
                                      growth_rate,
                                      filter_size,
                                      dropout_prob)

        self.dense = nn.Linear(embedding_dim + num_layers * growth_rate, hidden_dim)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(hidden_dim, labelset_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        embeddings = self.embedding(sequence)
        embeddings = embeddings.transpose(1, 2).unsqueeze(2).contiguous()

        feature_maps = self.dense_block(embeddings)
        feature_maps = feature_maps.squeeze(2).transpose(1, 2)

        feature_vec = self.dense(feature_maps).transpose(1, 2).max(-1)[0]
        feature_vec = self.relu(feature_vec)

        return self.softmax(self.proj(feature_vec))
