import numpy as np
from utils import load_mnist_data as load_mnist

def mean_squared_error(y, y_hat):
    return 0.5 * np.sum((y - y_hat)**2)

def cross_entropy_error(y, y_hat, one_hot=False):
    if y.dim == 1:
        y = y.reshape(1, y.size)
        y_hat = y_hat.reshape(1, y_hat.size)
    batch_size = y.shape[0]
    if one_hot:
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    else:
        return -np.sum(y_hat * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f, W, x, t):
    h = 1e-4 # rounding err 때문에 이 값이 제일 좋은 결과를 얻는다고 보고되고 있음
    grad = np.zeros_like(W) # x와 형상이 같은 배열을 생성
                            # simultaneously하게 연산해야함
    for idx in range(W.size):
        tmp_val = w[idx]

        W[idx] = tmp_val + h
        fxh1 = f(x, t)

        W[idx] = tmp_val - h
        fxh2 = f(x, t)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        W[idx] = tmp_val
    return grad

# def gradient_descent(f, init_x, lr=1e-2, step_num=100):
#     x = init_x
#     # Simultaneously하게 update
#     for i in range(step_num):
#         grad = numerical_gradient(f, x)
#         x -= lr * grad
#     return x

class simpleNet:

    def __init__(self):
        self.W = np.random.randn(2, 3) # normal distribution으로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

    @classmethod
    def f(cls, x, t):
        return cls.loss(x, t)


def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    net = simpleNet()
    x = np.array([0.6, 0.9])
    dW = numerical_gradient(net.f, net.W, x, t)
    print(dW)


if __name__ == '__main__':
    main()
