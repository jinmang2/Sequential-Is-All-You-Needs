from data import prepare_data

def main():
    train_iterator, valid_iterator, test_iterator, params = prepare_data()
    return train_iterator, valid_iterator, test_iterator, params

if __name__ == '__main__':
    a, b, c, params = main()
    print(params)
