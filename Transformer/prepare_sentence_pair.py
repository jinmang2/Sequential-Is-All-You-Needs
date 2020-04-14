from collections import Counter
import pandas as pd
import logging

from utils import time_decorator

def get_data_path(input_feed=False):
    if input_feed:
        response = input("아직은 excel파일만 읽어올 수 있습니다.")
        return response
    else:
        return "e:/aihub/2_대화체_200226.xlsx"

def read_data(data_path):
    df = pd.read_excel(data_path)
    return df

def make_field(SRC, TRG, src_tokenizer, trg_tokenizer):
    SRC_VOCAB = list(Counter([i for src in SRC for i in src_tokenizer(src)]).keys())
    TRG_VOCAB = list(Counter([i for trg in TRG for i in trg_tokenizer(trg)]).keys())

    SRC_VOCAB = ['<unk>', '<pad>', '<bos>', '<eos>'] + SRC_VOCAB
    TRG_VOCAB = ['<unk>', '<pad>', '<bos>', '<eos>'] + TRG_VOCAB

    SRC_VOCAB_i = list(range(len(SRC_VOCAB)))
    TRG_VOCAB_i = list(range(len(TRG_VOCAB)))

    SRC_VOCAB_DICT = {i:j for i,j in zip(SRC_VOCAB, SRC_VOCAB_i)}
    TRG_VOCAB_DICT = {i:j for i, j in zip(TRG_VOCAB, TRG_VOCAB_i)}

    return SRC_VOCAB_DICT, TRG_VOCAB_DICT

def spacing_tokenizer(s):
    return s.split(' ')

def sent2tokens(sents,
                vocab,
                tokenizer,
                init_token='<bos>',
                eos_token='<eos>',
                pad_token='<pad>',
                unk_token='<unk>'):
    bsz = len(sents)
    sents = list(map(tokenizer, sents))
    seg_length = [len(sent) for sent in sents]
    max_seq_len = max(seg_length) + 2 # BOS, EOS
    outputs = [[]] * bsz
    for ix, (sent, seg_len) in enumerate(zip(sents, seg_length)):
        output = [vocab[init_token]]
        output.extend([vocab.get(word, vocab[unk_token]) for word in sent])
        output.append(vocab[eos_token])
        if seg_len + 2 < max_seq_len: # BOS, EOS는 이미 추가
            output.extend(
                [vocab[pad_token]] * (max_seq_len - len(output)))
        outputs[ix] = output
    return outputs

def make_iterator(data, vocab, tokenizer,
                  batch_size=32,
                  init_token='<bos>',
                  eos_token='<eos>',
                  pad_token='<pad>',
                  unk_token='<unk>'):
    bsz = batch_size
    n_iter = len(data) // bsz
    for i in range(n_iter+1):
        batch_data = data[i*bsz:(i+1)*bsz] if i < n_iter else data[i*bsz:]
        if batch_data == []: break
        outputs = sent2tokens(batch_data, vocab, tokenizer)
        yield i, batch_data, outputs

@time_decorator
def get_seq_pair_kr_en(batch_size=64):
    data_path = get_data_path()
    df = read_data(data_path)
    tokenizer = spacing_tokenizer
    SRC = df.원문.tolist()
    TRG = df.번역문.apply(lambda x: x.lower()).tolist()
    SRC_VOCAB_DICT, TRG_VOCAB_DICT = make_field(SRC, TRG, tokenizer, tokenizer)
    src_generator = make_iterator(SRC, SRC_VOCAB_DICT, tokenizer, batch_size)
    trg_generator = make_iterator(TRG, TRG_VOCAB_DICT, tokenizer, batch_size)
    return src_generator, trg_generator
