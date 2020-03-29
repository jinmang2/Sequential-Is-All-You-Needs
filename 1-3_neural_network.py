import numpy as np

def step(x):
    return (x > 0).astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def init_network():
    # 네트워크는 가중치와 편향의 값을 내부에 저장하고 있다.
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 실제로 중간에 sigmoid를 활성화 함수로 사용하게 되면 vanishing gradient 문제에 직면하게 된다.
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(a1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def softmax(a, raise_overflow=False):
    c = np.max(a)
    if raise_overflow:
        exp_a = np.exp(c)
    else:
        exp_a = np.exp(c - a) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
    

def main():
    a = np.array([1010, 1000, 990])
    print(softmax(a, True))
    print(softmax(a))

if __name__ == '__main__':
    main()
