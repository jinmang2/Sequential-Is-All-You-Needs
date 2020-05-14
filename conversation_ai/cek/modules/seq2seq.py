import torch
import torch.nn as nn
import torch.nn.functional as F

from rhns import RecurrentHighwayNetwork


class Attention(nn.Module):
    pass


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hid_dim,
                 cell='LSTM',
                 num_highway=1,
                 rhn_recurrence_depth=5,
                 rhn_couple=False,
                 rhn_dropout_p=0.2,
                 rhn_init_weight='kaiming',
                 rhn_init_bias=-1,
                 dropout_p=0.2,):
        super(Encoder, self).__init__()

        if cell not in ['LSTM', 'GRU']:
            raise ValueError('Argument `cell` must be `LSTM` or `GRU`.')

        self.biRNN = getattr(nn, cell)(
            input_size=input_dim,
            hidden_size=hid_dim,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

        self.RHN = RecurrentHighwayNetwork(
            in_features=hid_dim,
            out_features=hid_dim,
            recurrence_depth=rhn_recurrence_depth,
            couple=rhn_couple,
            dropout_p=rhn_dropout_p,
            init_weight=rhn_init_weight,
            init_bias=rhn_init_bias,
        )

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
