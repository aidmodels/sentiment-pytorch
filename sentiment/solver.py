# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from mlpm.solver import Solver

import spacy
import torch
from torch.functional import split
import torchtext
from torchtext.legacy import data
import torch.nn as nn
import torch.nn.functional as F

MIN_LEN=7

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class SentimentSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        spacy.cli.download("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_sm")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('pretrained/imdb-model-cnn.pt')
        loaded_vectors = torchtext.vocab.Vectors('pretrained/glove.6B.100d.txt')
        self.TEXT = data.Field(lower=True, tokenize='spacy')
        self.Label = data.LabelField(dtype = torch.float)
        train = torchtext.datasets.IMDB(split="train")
        self.TEXT.build_vocab(train, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))
        self.TEXT.vocab.set_vectors(stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim)
        self.Label.build_vocab(train)
        self.model.eval()
        self.model = self.model.to(self.device)

    def forward_with_sigmoid(self, input):
        return torch.sigmoid(self.model(input))

    def infer(self, data):
        text = [tok.text for tok in self.nlp.tokenizer(data['text'].lower())]
        if len(text) < MIN_LEN:
                text += ['pad'] * (MIN_LEN - len(text))
        indexed = [self.TEXT.vocab.stoi[t] for t in text]
        self.model.zero_grad()
        input_indices = torch.tensor(indexed, device=self.device)
        input_indices = input_indices.unsqueeze(0)
        pred = self.forward_with_sigmoid(input_indices).item()
        return {"output": pred}
