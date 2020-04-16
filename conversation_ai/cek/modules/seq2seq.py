import torch
import torch.nn as nn


class Highway(nn.Module):

    def __init__(self,
                 input_dim,
                 num_layers=1,
                 activation=nn.functional.relu):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim*2)
             for _ in range(num_layers)]
        )
        self.activation = activation
        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_input = x
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self.input_dim]
            gate = projected_input[:, self.input_dim:self.input_dim*2]
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


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
        self.highway = Highway(, nn.ReLU())

    def forward(self, src):
        pass


class Decoder(nn.Module):
    pass


class Seq2Seq(nn.Module):
    pass
