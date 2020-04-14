import time
import random

def get_rand_ind(len_pairs, n):
    buffer = []
    for i in range(n):
        sample = random.choice(list(range(len_pairs)))
        while sample not in buffer:
            sample = random.choice(list(range(len_pairs)))
            buffer.append(sample)
    return buffer

def sampling(li, ind):
    output = []
    for i in ind:
        output.append(li[i])
    return output

def time_decorator(method):
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('\'{:s}\'  {:2.2f} ms  {:2.2f} sec  {:2.2f} min  {:2.2f} hour'.format(
                  method.__name__, (te - ts) * 1000, (te - ts), (te - ts) / 60, (te - ts) / 3600))
        return result
    return wrapper
