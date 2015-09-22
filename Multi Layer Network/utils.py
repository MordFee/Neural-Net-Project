import pickle
import gzip

from math import atan, exp

def heaviside(x): return max(x/abs(x), 0)

def sigmoid(x): return 1 / ( 1 + exp(-x))

def arctan(x): return atan(x)

def identity(x): return x

def positive(x): return max(x,0)

def addition(x, y): return x + y

def substraction(x, y): return x - y


def quick_load(path='mnist_14x14.pkl.gz'):
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    return train_set



