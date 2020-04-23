import torch
import torch.nn as nn


class Attention(nn.Module):
    pass


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hid_dim,
                 num_highway=1,
                 dropout_p=0.2,
                 bias_init=1,
                 activation=nn.functional.relu):
        super(Encoder, self).__init__()
        self.biRNN = nn.LSTM(input_size=input_dim,
                             hidden_size=hid_dim,
                             bias=True,
                             batch_first=True,
                             bidirectional=True)
        self.uniRNN = nn.LSTM(input_size=hid_dim,
                              hidden_size=hid_dim,
                              bias=True,
                              batch_first=True,
                              bidirectional=False)
        self.uniRNN2 = nn.LSTM(input_size=hid_dim,
                               hidden_size=hid_dim,
                               bias=True,
                               batch_first=True,
                               bidirectional=False)
        self.dropout = nn.Dropout(dropout_p)
        self.highway = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim*2, bias=True)
             for _ in range(num_highway)]
        )
        for layer in self.highway:
            layer.bias[hid_dim:].data.fill_(bias_init)
        self.activation = activation

    def forward(self, src):
        x, _ = self.biRNN(src)
        for T in self.highway:
            projected_input = layer(x)
            linear_part = x


        layer_x, _ = self.uniRNN(x)

        context, hidden_state = self.uniRNN2(x + layer_x)
        return context, hidden_state


class Decoder(nn.Module):
    pass


class Seq2Seq(nn.Module):
    pass
