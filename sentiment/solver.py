# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import spacy
from mlpm.solver import Solver

import torch
import torchtext
import torchtext.data
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vocab


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
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        model = torch.load('pretrained/imdb-model-cnn.pt')
        model.eval()
        model = model.to(device)


    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        sequences = self.loaded_tokenizer.texts_to_sequences([data['text']])
        padding = pad_sequences(sequences, maxlen=MAX_LEN)
        result = self.model.predict(padding, batch_size=1, verbose=1)
        return {"output": result.tolist()}  # return a dict
