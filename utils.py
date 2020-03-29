from dataset.mnist import load_mnist

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(normalize=True, one_hot_label=False)
    return (x_train, y_train), (x_test, y_test)
