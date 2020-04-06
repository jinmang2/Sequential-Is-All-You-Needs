import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        query = self.linear(h_t_tgt.squeeze(1)).unsqueeze(-1)
        weight = torch.bmm(h_src, query).squeeze(-1)
        if mask in not None:
            weight.masked_fill_(mask, -float("inf"))
        weight = self.softmax(weight)
        context_vector = torch.bmm(weight.unsqueeze(1), h_src)
        return context_vector


class Encoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=0.2):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(word_vec_dim,
                           int(hidden_size / 2),
                           num_layers=n_layers,
                           dropout=dropout_p,
                           bidirectional=True,
                           batch_first=True)

    def forward(self, emb):
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = emb
        y, h = self.rnn(x)
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)
        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=0.2):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(word_vec_dim + hidden_size,
                           hidden_size,
                           num_layers=n_layers,
                           dropout=dropout_p,
                           bidirectional=False,
                           batch_first=True)

    def forward(self, emb_t, g_t_1_tilde, h_t_1):
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)
        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)
        y, h = self.rnn(x, h_t_1)
        return y, h


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        y = self.softmax(self.output(x))
        return y


class Seq2Seq(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 output_size,
                 n_layers=4,
                 dropout_p=0.2):
        self.input_size = input_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_dim)
        self.emb_dec = nn.Embedding(output_size, word_vec_dim)

        self.encoder = Encoder(word_vec_dim,
                               hidden_size,
                               n_layers=n_layers,
                               dropout_p=dropout_p)
        self.decoder = Decoder(word_vec_dim,
                               hidden_size,
                               n_layers=n_layers,
                               dropout_p=dropout_p)
        self.attn = Attention(hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length):
        mask = []
        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                   ], dim=1)]
            else:
                mask += [x.new_ones(1, l).zero_()]
        mask = torch.cat(mask, dim=0).byte()
        return mask

    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []
        hiddens, cells = encoder_hiddens
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i+1]], dim=-1)]
            new_cells += [torch.cat([cells[i], cells[i+1]], dim=-1)]
        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
        return (new_hiddens, new_cells)

    def forward(self, src, tgt):
        batch_size = tgt.size(0)
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        if isinstance(tgt, tuple):
            tgt = tgt[0]
        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size,
                                                            ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size,
                                                            ).transpose(0, 1).contiguous()
        h_0_tgt = (h_0_tgt, c_0_tgt)
        emb_tgt = self.emb_dec(tgt)
        h_tilde = []
        h_t_tilde = None
        decoder_hidden = h_0_tgt
        for t in range(tgt.size(1)):
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden)
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector],
                                                        dim=-1)))
            h_tilde += [h_t_tilde]
        h_tilde = torch.cat(h_tilde, dim=1)
        y_hat = self.generator(h_tilde)
        return y_hat
