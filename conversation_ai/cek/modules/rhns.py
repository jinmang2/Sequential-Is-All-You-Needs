import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mylinear import MyLinear as Linear


class RNNDropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        shape = (x.size(0), 1, x.size(2))
        m = self.dropout_mask(x.data, shape, self.p)
        return x * m

    @staticmethod
    def dropout_mask(x, sz, p):
        return x.new(*sz).bernoulli_(1-p).float().div_(1-p)


class HighwayBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 first=False,
                 couple=False,
                 dropout_p=0.0,
                 init_weight='kaiming',
                 init_bias=-1):
        super().__init__()
        self.first = first
        self.couple = couple
        if first:
            self.W_H = Linear(in_features, out_features, bias=False, activation=None)
            self.W_T = Linear(in_features, out_features, bias=False, activation=None)
            if not couple:
                self.W_C = Linear(in_features, out_features, bias=False, activation=None)
        self.R_H = Linear(in_features, out_features, bias=True, activation=None)
        self.R_T = Linear(in_features, out_features, bias=True, activation=None)
        if not couple:
            self.R_C = Linear(in_features, out_features, bias=True, activation=None)
        for child in self.children():
            child.reset_parameters(init_weight, init_bias)
        self.dropout = RNNDropout(dropout_p)

    def forward(self, x, s):
        if self.first:
            h = torch.tanh(self.W_H(x) + self.R_H(s))
            t = torch.sigmoid(self.W_T(x) + self.R_T(s))
            if self.couple:
                c = 1 - t
            else:
                c = torch.sigmoid(self.W_C(x) + self.R_C(s))
        else:
            h = torch.tanh(self.R_H(s))
            t = torch.sigmoid(self.R_T(s))
            if self.couple:
                c = 1 - t
            else:
                c = torch.sigmoid(self.R_C(s))
        t = self.dropout(t.unsqueeze(0)).squeeze(0)
        return h * t + s * c


class RecurrentHighwayNetwork(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 recurrence_depth=5,
                 couple=False,
                 dropout_p=0.,
                 init_weight='kaiming',
                 init_bias=-1):
        super().__init__()
        self.highways = nn.ModuleList(
            [
                HighwayBlock(
                    in_features, 
                    out_features,
                    first=True if l == 0 else False,
                    couple=couple,
                    dropout_p=dropout_p,
                    init_weight=init_weight,
                    init_bias=init_bias
                )
                for l in range(recurrence_depth)
            ]
        )
        self.recurrence_depth = recurrence_depth
        # self.hidden_dim = out_features

    def forward(self, input, hidden):
        # Expects input dimensions [seq_len, bsz, input_dim]
        outputs = []
        for x in input:
            for block in self.highways:
                hidden = block(x, hidden)
            outputs.append(hidden)
        outputs = torch.stack(outputs)
        return outputs, hidden


print(getattr(nn, 'LSTM'))