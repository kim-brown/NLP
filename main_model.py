from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class MainModel(nn.Module):
    def __init__(self, vocab_size):
        """

        """
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, 1)

    def forward(self, input):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)

        :return: the logits, a tensor of shape
                 (batch_size, seq_len, vocab_size)
        """
        out1 = self.linear1(input)
        s = nn.Softmax(dim=0)
        return s(out1)
