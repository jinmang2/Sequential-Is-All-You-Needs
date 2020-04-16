import torch
import torch.nn as nn


class Attention(nn.Module):
    pass


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hid_dim):
        super(Encoder, self).__init__()
        params = {
            'input_size' : input_dim,
            'hidden_size': hid_dim,
            'bias'       : True,
            'batch_first': True,
        }
        self.biRNN = nn.LSTM(**params, bidirectional=True)
        self.uniRNN = nn.LSTM(**params, bidirectional=False)
        self.gate_layer = nn.Linear(hid_dim, hid_dim, bias=True)

    def forward(self, src):
        output, hidden_state = self.biRNN(src)



class Decoder(nn.Module):
    pass


class Seq2Seq(nn.Module):
    pass
