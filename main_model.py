from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class MainModel(nn.Module):
    def __init__(self, vocab_size, hidden_layer_size, embedding_size, window_size, num_layers):
        """

        """
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_layer_size,
        num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_layer_size, 1)
        self.linear2 = nn.Linear(window_size, 1)

    def forward(self, input):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)

        :return: the logits, a tensor of shape
                 (batch_size, 1)
        """
        embs = self.embeddings(input)
        out1, _ = self.lstm(embs)
        out2 = self.linear1(out1).squeeze()
        s = nn.Softmax(dim=0)
        out3 = self.linear2(out2)
        return s(out3)
