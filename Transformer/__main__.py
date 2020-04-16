# https://mrcoding.tistory.com/entry/아톰에서-파이썬-스크립트-실행시-한글-깨짐현상-잡는-꿀팁
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

from prepare_sentence_pair import get_seq_pair_kr_en as get_generator
from transformer import Transformer

BATCH_SIZE = 64

if __name__ == '__main__':
    src_generator, trg_generator = get_generator(batch_size=BATCH_SIZE)
    _, _, src_input = next(src_generator)
    _, _, trg_input = next(trg_generator)
    print(src_input, trg_input)
    print(Transformer)
