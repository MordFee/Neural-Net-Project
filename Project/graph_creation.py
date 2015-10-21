import numpy as np

def pseudo_random_square_graph_1( n=int, p=int):
    """
    Returns a n*n binary matrix.
    Where each node has p edges from entryIndex to entryIndex + p
    """
    if p >=n: # This is fully connected
        return np.ones((n, n))
    maskMatrix = np.zeros( (n,n))
    for i in range(n):
        for j in range(i, p+i):
            k = j % n
            maskMatrix[i][k] = 1
    return maskMatrix

def pseudo_random_square_graph_2( n=int, proba=list):
    """
    Returns a n*n binary matrix.
    proba is a list of n floats between 0 and 1.
    Each node i has the probability proba[i] to have an edge node j
    """
    maskMatrix = np.zeros( (n,n))
    for i in range(n):
        for j in range(n):
            r = np.random.random()
            if r <= proba[i]:
                maskMatrix[i][j] = 1
    return maskMatrix


def pseudo_random_rect_graph_1( n=int, m=int, p=int):
    #TODO Check that it is n inputs & m outputs and not the opposite
    """
    Returns a n input, m output binary matrix.
    Where each node has p edges from entryIndex to entryIndex + p
    """
    if p >= m: # This is fully connected
        return np.ones( (n, m))
    maskMatrix = np.zeros( (n, m))
    for i in range(n):
        for j in range(i, p+i):
            k = j % m
            maskMatrix[i][k] = 1
    return maskMatrix

def pseudo_random_rect_graph_2( n=int, m=int, proba=list):
    #TODO Check that it is n inputs & m outputs and not the opposite
    """
    Returns a n input, m output binary matrix.
    proba is a list of n floats between 0 and 1.
    Each node i has the probability proba[i] to have an edge node j
    """
    maskMatrix = np.zeros( (n,m))
    for i in range(n):
        for j in range(m):
            r = np.random.random()
            if r <= proba[i]:
                maskMatrix[i][j] = 1
    return maskMatrix

# aliases
pseudorect2 = pseudo_random_rect_graph_2
pseudorect1 = pseudo_random_rect_graph_1
pseudosquare2 = pseudo_random_square_graph_2
pseudosquare1 = pseudo_random_square_graph_1

from keraspatal.utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True, kwargs=kwargs)


if __name__=='__main__':

    print(pseudo_random_square_graph_1(10,3))
    #print(pseudo_random_square_graph_2(10,[1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 1, 0.5]))

    #print(pseudo_random_rect_graph_1(10, 5, 3))
    #print(pseudo_random_rect_graph_2(10, 5, [1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 1, 0.5]))