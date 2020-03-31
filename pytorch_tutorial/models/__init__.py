__all__ = ['Attention', 'RNNencdec', 'Transformer']

from modules.attention import Attention

if __name__ == '__main__':
    print(Attention(10, 10))
