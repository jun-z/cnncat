import sys
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from spacy.lang.en import EnglishDefaults

tokenizer = EnglishDefaults.create_tokenizer()


def get_logger(name, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    stdout_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('[cnncat][%(name)s]'
                                  '[%(asctime)s][%(levelname)s]:%(message)s')

    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.setLevel(level)

    return logger


def log_args(logger: logging.Logger, args):
    for k, v in vars(args).items():
        logger.info(f'--{k}:{v}')


def tokenize(text):
    return [token.lower_ for token in tokenizer(text)]


def pad_shorties(x, min_len, pad_token='<pad>'):
    if len(x) < min_len:
        return x + [pad_token] * (min_len - len(x))
    else:
        return x


def split_data(dataset: data.TabularDataset,
               fields,
               random_seed=1123,
               valid_split=.2):

    random.seed(1234)

    train_examples = []
    valid_examples = []
    for example in dataset.examples:
        rn = random.random()
        if rn < valid_split:
            valid_examples.append(example)
        else:
            train_examples.append(example)

    train_set = data.Dataset(train_examples, fields)
    valid_set = data.Dataset(valid_examples, fields)
    return train_set, valid_set


def calc_loss(output, labels):
    return nn.functional.nll_loss(output, labels, size_average=False).data[0]


def calc_entropy(log_probs):
    return torch.mean(-torch.sum(torch.exp(log_probs) * log_probs, -1))


def predict(output):
    return output.data.max(1)[1]


def evaluate(dataset: data.Dataset,
             batch_size: int,
             classifier: nn.Module,
             device):

    classifier.eval()

    iterator = data.BucketIterator(dataset,
                                   batch_size,
                                   lambda x: len(x.text),
                                   device,
                                   train=False)

    loss = 0
    correct = 0
    for batch in iterator:
        output = classifier(batch.text)
        preds = predict(output)

        loss += calc_loss(output, batch.label)
        correct += preds.eq(batch.label.data).cpu().sum()

    return loss / len(dataset), correct / len(dataset)
