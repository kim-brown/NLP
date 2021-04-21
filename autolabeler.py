from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class AutoLabeler(nn.Module):
    def __init__(self, vocab_size, hidden_layer_size):
        """

        """
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, 1)
        # self.linear3 = nn.Linear(hidden_layer_size, 1)

    def forward(self, input):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)

        :return: the logits, a tensor of shape
                 (batch_size, 1)
        """
        out1 = self.linear1(input)
        out2 = self.linear2(out1)
        s = nn.Softmax(dim=0)
        # out3 = self.linear3(out2)
        return s(out2)
