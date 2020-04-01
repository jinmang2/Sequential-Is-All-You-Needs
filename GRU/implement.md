# GRU implementation

**First**, import `torch`
```python
import torch
import torch.nn as nn # torch's neural network
```

**Second**, Set hyperparameters
```python
SEQ_LEN = 10                       # Length of Sequence
BATCH_SIZE = 3                     # Number of Batch Size
INPUT_DIM = 30                     # Number of tokens in vocabulary
ENC_EMB_DIM = DEC_EMB_DIM = 32     # Embedding Dimension
ENC_HID_DIM = DEC_HID_DIM = 64     # Hidden Dimension
ENC_DROPOUT = DEC_DROPOUT = 0.5    # Dropout ratio
```

**Third**, Let sample inputs `x` as follow;
```python
# Since input is natural language,
# Token in $\mathbb{N}$
x = torch.randint(0+1, INPUT_DIM-2, size=(SEQ_LEN, BATCH_SIZE))
x[ 0, :] = 0                # <BOS>, Begin of Sentence
x[-1, :] = INPUT_DIM - 1    # <EOS>, End of Sentence

print(x, x.shape)
```
```
tensor([[ 0,  0,  0],
        [ 9, 19, 22],
        [23, 24, 10],
        [22, 18,  8],
        [20,  8, 26],
        [12, 12, 13],
        [14,  1, 24],
        [11, 18, 14],
        [12, 22, 12],
        [29, 29, 29]]) torch.Size([10, 3])
```

**Fourth**, Embedding `x` to `embedded`
```python
embedding = nn.Embedding(INPUT_DIM, ENC_EMB_DIM)
embedding.weight.shape
>>> torch.Size([30, 32])
# number of token is 30.
# For each token, token in $\mathbb{N}$
# Using Embedding layer, map token in $\mathbb{N}$ to vector in $\mathbb{R}^{ENC_EMB_DIM}$

embedded = embedding(x)
x.shape # input size is,
>>> torch.Size([10, 3])
embedded.shape # embedded size is,
>>> torch.Size([10, 3, 32])
```

**Fifth**, Define GRU hidden unit
```python
rnn = nn.GRU(ENC_EMB_DIM, ENC_HID_DIM)
# `input_size`: The number of expected features in the input `x`
# `hidden_size`: The number of features in the hidden state `h`
# `num_layers`: Number of recurrent layers.
#     Default is 1
# `bias`: If False, then the layer does not use bias weights `b_ih` and `b_hh`.
#     Default is True
# `batch_first`: If True, then the input and output tensors are provided
#     as (batch, seq, features)
#     Default is False
# `dropout`: dropout ratio
#     Default is 0
# `bidirectional`: If True, becomes a bidirectional GRU
#     Default is False
```

**Sixth**, Get `output` and `hidden` from rnn
```python
output, hidden = rnn(embedded)
print(output.shape, hidden.shape)
>>> torch.Size([10, 3, 64]), torch.Size([1, 3, 64])
```

How can we get the above result?

Let's check the source code of `torch.nn.module.rnn.py`

```python
[573] class GRU(RNNBase):
[...]     ...
[677]     def __init__(self, *args, **kwargs):
[678]         super(GRU, self).__init__('GRU', *args, **kwargs)
[...]     ...
[690]     def forward(self, input, hx=None):  # noqa: F811
[691]         orig_input = input
[692]         # xxx: isinstance check needs to be in conditional for TorchScript to compile
[693]         if isinstance(orig_input, PackedSequence):
[694]             input, batch_sizes, sorted_indices, unsorted_indices = input
[695]             max_batch_size = batch_sizes[0]
[696]             max_batch_size = int(max_batch_size)
[697]         else:
[698]             batch_sizes = None
[699]             max_batch_size = input.size(0) if self.batch_first else input.size(1)
[700]             sorted_indices = None
[701]             unsorted_indices = None
[702] 
[703]         if hx is None:
[704]             num_directions = 2 if self.bidirectional else 1
[705]             hx = torch.zeros(self.num_layers * num_directions,
[706]                              max_batch_size, self.hidden_size,
[707]                              dtype=input.dtype, device=input.device)
[708]         else:
[709]             # Each batch of the hidden state should match the input sequence that
[710]             # the user believes he/she is passing in.
[711]             hx = self.permute_hidden(hx, sorted_indices)
[712] 
[713]         self.check_forward_args(input, hx, batch_sizes)
[714]         if batch_sizes is None:
[715]             result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
[716]                              self.dropout, self.training, self.bidirectional, self.batch_first)
[717]         else:
[718]             result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
[719]                              self.num_layers, self.dropout, self.training, self.bidirectional)
[720]         output = result[0]
[721]         hidden = result[1]
[722] 
[723]         # xxx: isinstance check needs to be in conditional for TorchScript to compile
[724]         if isinstance(orig_input, PackedSequence):
[725]             output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
[726]             return output_packed, self.permute_hidden(hidden, unsorted_indices)
[727]         else:
[728]             return output, self.permute_hidden(hidden, unsorted_indices)
```

Pay attention to line 714.

This is where torch calculates `GRU` with the c++ imperative engine.

Then, what is `torch._VF`?
```python
# torch._VF.py

import torch
import sys
import types


class VFModule(types.ModuleType):
    def __init__(self, name):
        super(VFModule, self).__init__(name)
        self.vf = torch._C._VariableFunctions
        
    def __getattr__(self, attr):
        return getattr(self.vf, attr)
        
sys.modules[__name__] = VFModule(__name__)
```

VF is Abbreviation of `VariableFunctions`

Python's `__getattr__` magic command handles the case of attempting to access an attribute on the object.

In this case, `VFModule.gru` is equal to `VFModule.vf.gru`.

Since `VFModule.vf` is assigned by `torch._C._VariableFunctions`, This command calls the imperative engine of C ++ from `torch._C._VariableFunctions.gru`

Let's leave implementation of c++ as a future task, and let's implement it in pythonic concept.

In [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
](https://arxiv.org/pdf/1406.1078.pdf) papers, GRU is debribed by

![img](https://miro.medium.com/max/1400/1*6eNTqLzQ08AABo-STFNiBw.png)
ref. https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

GRU has 2 gateway.

**reset gate**
![img](https://miro.medium.com/max/1400/1*j1j1mLIyTm97hCay4GRC_Q.png)

**update gate**
![img](https://miro.medium.com/max/1400/1*o7NzuF8w0H7qybG8Fn-Shw.png)

Then, the hidden state of each time step is updated by,

**current memory content**
![img](https://miro.medium.com/max/1400/1*CxQBMqy8dvgJNjeJcur6pQ.png)

**Final memory**
![img](https://miro.medium.com/max/1400/1*zxSTnqedwLRoicgHKYKsVQ.png)

Using above equation, let's implement it :)

**Seventh**, get a parameters
```python
# num_layers and num_directions are `1`
num_layers = rnn.num_layers
num_directions = 2 if rnn.bidirectional else 1

w_ih, w_hh, b_ih, b_hh = list(rnn.parameters())

# GRU has 3 paremeters. so, `GATE_SIZE` is 3.
w_ih.shape, w_hh.shape, b_ih.shape, b_hh.shape
>>> (torch.Size([192, 32]),  # w_ih, [GATE_SIZE * HID_DIM, EMB_DIM]
>>>  torch.Size([192, 64]),  # w_hh, [GATE_SIZE * HID_DIM, HID_DIM]
>>>  torch.Size([64]),       # b_ih, [GATE_SIZE * HID_DIM,]
>>>  torch.Size([64]))       # b_hh, [GATE_SIZE * HID_DIM,]

# Since Wr, Wz, W is flattend, let's hand it out!
Wr_ih = w_ih[:ENC_HID_DIM,:]
Wz_ih = w_ih[ENC_HID_DIM:2*ENC_HID_DIM,:]
W_ih  = w_ih[2*ENC_HID_DIM:,:]

Wr_hh = w_hh[:ENC_HID_DIM,:]
Wz_hh = w_hh[ENC_HID_DIM:2*ENC_HID_DIM,:]
W_hh  = w_hh[2*ENC_HID_DIM:,:]

br_ih = b_ih[:ENC_HID_DIM]
bz_ih = b_ih[ENC_HID_DIM:2*ENC_HID_DIM]
b_ih  = b_ih[2*ENC_HID_DIM:]

br_hh = b_hh[:ENC_HID_DIM]
bz_hh = b_hh[ENC_HID_DIM:2*ENC_HID_DIM]
b_hh  = b_hh[2*ENC_HID_DIM:]

Wr_ih.shape, Wz_ih.shape, W_ih.shape
>>> (torch.Size([64, 32]), torch.Size([64, 32]), torch.Size([64, 32]))

Wr_hh.shape, Wz_hh.shape, W_hh.shape
>>> (torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64]))

br_ih.shape, bz_ih.shape, b_ih.shape
>>> (torch.Size([64]), torch.Size([64]), torch.Size([64]))

br_hh.shape, bz_hh.shape, b_hh.shape
>>> (torch.Size([64]), torch.Size([64]), torch.Size([64]))

output_ = []
# Generate initial hidden state
hx = torch.zeros(num_layers * num_directions,
                 BATCH_SIZE, ENC_HID_DIM)
for i in range(SEQ_LEN):
    token_embedding = embedded[i, :, :]
    # reset gate
    left_term   = token_embedding.unsqueeze(0).matmul(Wr_ih.permute(1,0)) + br_ih
    right_term  = hx.matmul(Wr_hh.permute(1,0)) + br_hh
    reset_gate  = torch.sigmoid(left_term + right_term)
    # update gate
    left_term   = token_embedding.unsqueeze(0).matmul(Wz_ih.permute(1,0)) + bz_ih
    right_term  = hx.matmul(Wz_hh.permute(1,0)) + bz_hh
    update_gate = torch.sigmoid(left_term + right_term)
    # Current memory content
    left_term   = token_embedding.unsqueeze(0).matmul(W_ih.permute(1,0)) + b_ih
    right_term  = (reset_gate * hx).matmul(W_hh) + b_hh
    h_hat       = torch.tanh(left_term + right_term)
    # Final memory at current time step
    hx          = update_gate * hx + (1-update_gate) * h_hat
    output_.append(hx)
output_ = torch.cat(output_, dim=0)
```

### Compare `torch`'s GRU and My GRU
```python
# torch's GRU result
output
>>> tensor([[[-0.0087, -0.3379, -0.1505,  ..., -0.1969,  0.0905,  0.3108],
>>>          [-0.0087, -0.3379, -0.1505,  ..., -0.1969,  0.0905,  0.3108],
>>>          [-0.0087, -0.3379, -0.1505,  ..., -0.1969,  0.0905,  0.3108]],
>>> 
>>>         [[ 0.1175,  0.2644, -0.4740,  ...,  0.0846,  0.4866,  0.0589],
>>>          [-0.1936,  0.0840, -0.1545,  ...,  0.0013,  0.0537,  0.3784],
>>>          [-0.2266,  0.1992, -0.1174,  ..., -0.0499,  0.1098,  0.0057]],
>>> 
>>>         [[-0.0384, -0.0441, -0.1637,  ..., -0.0986, -0.0969,  0.3520],
>>>          [-0.2901,  0.4000, -0.3992,  ...,  0.2688, -0.2010,  0.1623],
>>>          [ 0.1503,  0.3025, -0.0056,  ...,  0.1264,  0.2715, -0.0238]],
>>> 
>>>         ...,
>>> 
>>>         [[-0.6688,  0.4954, -0.3371,  ..., -0.0798, -0.3500,  0.3815],
>>>          [-0.2476,  0.4638, -0.2258,  ...,  0.3591, -0.3438, -0.1727],
>>>          [ 0.2113, -0.1116,  0.0288,  ..., -0.0676, -0.1760,  0.2817]],
>>> 
>>>         [[-0.5959,  0.4002, -0.0354,  ..., -0.0341, -0.0722,  0.4735],
>>>          [-0.2377,  0.4763, -0.1615,  ...,  0.4727, -0.4981, -0.3504],
>>>          [ 0.2241, -0.1988,  0.2976,  ..., -0.0341, -0.3548,  0.1504]],
>>> 
>>>         [[-0.0541,  0.3023,  0.2694,  ...,  0.2724, -0.1082, -0.0695],
>>>          [ 0.1278,  0.3538,  0.2153,  ...,  0.4598, -0.3654, -0.3590],
>>>          [ 0.3327, -0.0788,  0.3787,  ...,  0.2239, -0.3053, -0.0856]]],
>>>        grad_fn=<StackBackward>)

# My GRU result
output_
>>> tensor([[[-0.0316, -0.3213, -0.1689,  ..., -0.1971,  0.0838,  0.3075],
>>>          [-0.0316, -0.3213, -0.1689,  ..., -0.1971,  0.0838,  0.3075],
>>>          [-0.0316, -0.3213, -0.1689,  ..., -0.1971,  0.0838,  0.3075]],
>>> 
>>>         [[ 0.0553,  0.2746, -0.4742,  ...,  0.0328,  0.5028,  0.0315],
>>>          [-0.2428,  0.1107, -0.1433,  ..., -0.0369,  0.0901,  0.3416],
>>>          [-0.2981,  0.2128, -0.1277,  ..., -0.0907,  0.1494, -0.0319]],
>>> 
>>>         [[-0.0626, -0.0760, -0.1934,  ..., -0.1095, -0.0719,  0.3484],
>>>          [-0.2682,  0.3816, -0.4158,  ...,  0.2071, -0.0428,  0.1982],
>>>          [ 0.1353,  0.3402, -0.1485,  ...,  0.1137,  0.2782, -0.0238]],
>>> 
>>>         ...,
>>> 
>>>         [[-0.6266,  0.4583, -0.4754,  ..., -0.0932, -0.3099,  0.3952],
>>>          [-0.2284,  0.4190, -0.3593,  ...,  0.3812, -0.2960,  0.0038],
>>>          [ 0.0484, -0.1496,  0.0408,  ..., -0.0838, -0.0977,  0.2543]],
>>> 
>>>         [[-0.5071,  0.3423, -0.2474,  ..., -0.0291, -0.0256,  0.4767],
>>>          [-0.2196,  0.3847, -0.3739,  ...,  0.4704, -0.4174, -0.2086],
>>>          [ 0.0253, -0.1454,  0.3362,  ..., -0.0432, -0.3355,  0.1270]],
>>> 
>>>         [[ 0.0393,  0.2219,  0.0647,  ...,  0.2568, -0.0283, -0.0191],
>>>          [ 0.1553,  0.2571, -0.0238,  ...,  0.4543, -0.2940, -0.2126],
>>>          [ 0.2354,  0.0079,  0.3905,  ...,  0.2186, -0.2967, -0.1592]]],
>>>        grad_fn=<CatBackward>)
```

### Visualization
```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(30, 8))
plt.plot(output.flatten().detach().numpy())
plt.plot(output_.flatten().detach().numpy())
plt.show()
```
![img1](https://user-images.githubusercontent.com/37775784/78113095-3091ad00-743a-11ea-9a4e-d57a28dcdf86.png)

