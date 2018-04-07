import re
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data

from utils import helpers
from model import CNNClassifier


# Command-line arguments.
parser = argparse.ArgumentParser(description='Train a model.')

parser.add_argument(
    '--train_file', required=True, help='training file path')

parser.add_argument(
    '--valid_split', default=.2, type=float, help='split for validation set')

parser.add_argument(
    '--random_seed', default=1234, type=int, help='random seed for splitting')

parser.add_argument(
    '--token_regex', default='\w+', help='tokenizing regex')

parser.add_argument(
    '--min_freq', default=5, type=int, help='min frequency for vocab')

parser.add_argument(
    '--num_epochs', default=10, type=int, help='number of epochs')

parser.add_argument(
    '--batch_size', default=128, type=int, help='batch size')

parser.add_argument(
    '--learning_rate', default=1e-3, type=float, help='learning rate')

parser.add_argument(
    '--threshold', default=1e-4, type=float, help='threshold for comparing loss')

parser.add_argument(
    '--patience', default=3, type=int, help='patience for learning rate decay')

parser.add_argument(
    '--decay_factor', default=.5, type=float, help='decay factor for learning rate')

parser.add_argument(
    '--logging_interval', default=100, type=int, help='logging interval')

parser.add_argument(
    '--embedding_dim', default=300, type=int, help='embedding dimension')

parser.add_argument(
    '--pretrained_embeddings', type=str, help='pretrained embeddings')

parser.add_argument(
    '--hidden_dim', default=100, type=int, help='hidden dimension')

parser.add_argument(
    '--num_layers', default=5, type=int, help='number of layers')

parser.add_argument(
    '--growth_rate', default=32, type=int, help='growth rate')

parser.add_argument(
    '--filter_size', default=3, type=int, help='filter size')

parser.add_argument(
    '--dropout_prob', default=.5, type=float, help='dropout probability')

parser.add_argument(
    '--disable_cuda', action='store_true', help='disable cuda')

parser.add_argument(
    '--device_id', default=0, type=int, help='id for cuda device')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


# Training function.
def train():
    # Logger.
    logger = helpers.get_logger('training')

    helpers.log_args(logger, args)

    # Prepare training and testing data.
    WORD = re.compile(args.token_regex)

    TEXT = data.Field(lower=True,
                      tokenize=WORD.findall,
                      batch_first=True)

    LABEL = data.Field(sequential=False)

    fields = [
        ('label', LABEL),
        ('text', TEXT)
    ]

    train_set = data.TabularDataset(args.train_file, 'csv', fields)

    logger.info(f'Loaded training data: {args.train_file}')

    TEXT.build_vocab(train_set,
                     min_freq=args.min_freq,
                     vectors=args.pretrained_embeddings)

    LABEL.build_vocab(train_set)

    train_set, valid_set = helpers.split_data(train_set,
                                              fields,
                                              args.random_seed,
                                              args.valid_split)

    logger.info(f'Number of training examples: {len(train_set.examples)}')
    logger.info(f'Number of validation examples: {len(valid_set.examples)}')
    logger.info(f'Size of vocabulary: {len(TEXT.vocab)}')
    logger.info(f'Number of labels: {len(LABEL.vocab)}')

    # Initiate criterion, classifier, and optimizer.
    classifier = CNNClassifier(vocab_size=len(TEXT.vocab),
                               labelset_size=len(LABEL.vocab),
                               embedding_dim=args.embedding_dim,
                               hidden_dim=args.hidden_dim,
                               num_layers=args.num_layers,
                               growth_rate=args.growth_rate,
                               filter_size=args.filter_size,
                               dropout_prob=args.dropout_prob,
                               pretrained_embeddings=TEXT.vocab.vectors)

    if args.cuda:
        classifier.cuda(device=args.device_id)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), args.learning_rate)

    iterator = data.BucketIterator(dataset=train_set,
                                   batch_size=args.batch_size,
                                   sort_key=lambda x: len(x.text),
                                   device=args.device_id if args.cuda else -1)

    patience = args.patience
    min_valid_loss = None
    for batch in iterator:
        optimizer.zero_grad()
        loss = criterion(classifier(batch.text), batch.label)
        loss.backward()
        optimizer.step()

        progress, epoch = math.modf(iterator.epoch)

        if iterator.iterations % args.logging_interval == 0:
            valid_loss, accuracy = helpers.evaluate(valid_set,
                                                    args.batch_size,
                                                    classifier,
                                                    args.device_id if args.cuda else -1)

            logger.info(f'Epoch {int(epoch):<2} | '
                        f'progress: {progress:<6.2%} | '
                        f'training loss: {loss.data[0]:6.4f} | '
                        f'validation loss: {valid_loss:6.4f} | '
                        f'validation accuracy: {accuracy:<6.2%} |')

            classifier.train()

            if min_valid_loss is None:
                min_valid_loss = valid_loss

            if valid_loss < min_valid_loss + args.threshold:
                patience = args.patience
                min_valid_loss = min(valid_loss, min_valid_loss)
            else:
                patience -= 1
                if patience == 0:
                    logger.info(f'Patience of {args.patience} reached, decaying learning rate')
                    helpers.decay_learning_rate(optimizer, args.decay_factor)
                    patience = args.patience

        if epoch == args.num_epochs:
            break


if __name__ == '__main__':
    train()
