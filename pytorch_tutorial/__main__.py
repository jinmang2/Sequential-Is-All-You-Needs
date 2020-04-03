from data import prepare_data

from models.RNNencdec import Seq2Seq as RNN_Seq2Seq
from models.RNNencdec import Encoder as RNN_Seq2Seq_ENC
from models.RNNencdec import Decoder as RNN_Seq2Seq_DEC

from models.Attention import Seq2Seq as Attention
from models.Attention import Encoder as Attention_ENC
from models.Attention import Decoder as Attention_DEC
from models.modules.attention import Attention as Attention_attn

from models.Transformer import Seq2Seq as Transformer
from models.Transformer import Encoder as Transformer_ENC
from models.Transformer import Decoder as Transformer_DEC


def main(modelname=None):
    if modelname not in ['RNNSeq2Seq', 'Attention', 'Transformer']:
        raise TypeError("model name must be one of these three, "
                        "(RNNSeq2Seq, Attention, Transformer)")
    train_iterator, valid_iterator, test_iterator, params = prepare_data()
    (INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM,
     ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, DEC_DROPOUT,
     SRC_PAD_IDX, TRG_PAD_IDX) = params
    # INPUT_DIM = 7855
    # OUTPUT_DIM = 5893
    # ENC_EMB_DIM = 256
    # DEC_EMB_DIM = 256
    # ENC_HID_DIM = 512
    # DEC_HID_DIM = 512
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if modelname == 'RNNSeq2Seq':
        enc = RNNSeq2Seq_ENC(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
        dec = RNNSeq2Seq_DEC(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
        model = RNNSeq2Seq(enc, dec, device).to(device)
    elif modelname == 'Attention':
        attn = Attention_Attn(ENC_HID_DIM, DEC_HID_DIM)
        enc = Attention_ENC(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Attention_DEC(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
        model = Attention(enc, dec, device).to(device)
    elif modelname == 'Transformer':
        enc = Transformer_ENC(INPUT_DIM,
                              HID_DIM,
                              ENC_LAYERS,
                              ENC_HEADS,
                              ENC_PF_DIM,
                              ENC_DROPOUT,
                              device)
        dec = Transformer_DEC(OUTPUT_DIM,
                              HID_DIM,
                              DEC_LAYERS,
                              DEC_HEADS,
                              DEC_PF_DIM,
                              DEC_DROPOUT,
                              device)


if __name__ == '__main__':
