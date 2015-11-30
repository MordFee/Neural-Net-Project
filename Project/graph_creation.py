import numpy as np
import random

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


def fibonacci_sparse_matrix(n=int, m=int, k=int):
    mask_matrix = np.zeros( (n,m))
    f_vector = get_fibonacci_vector(m, k)
    for i in range(n):
        for j in range(m):
            mask_matrix[i][j] = f_vector[(j-i)%m]
    return mask_matrix

def get_fibonacci_vector(m=int, k=int):
    f_vector = np.zeros(m)
    if k == 1:
        f_vector[0] = 1
    else:
        f_list = [1,2]
        while len(f_list) < k:
            f_list.append(f_list[-1]+f_list[-2])
        if f_list[-1] <= m:
            for f in f_list:
                f_vector[f-1] = 1
        else:
            f_list = map(lambda(a): a*(m-1)/f_list[-1], f_list)
            for f in f_list:
                while f_vector[f] == 1:
                    f += 1
                f_vector[f] = 1
    return f_vector

def long_short_sparse_matrix(n=int, m=int, k=int):
    mask_matrix = np.zeros( (n,m))
    f_vector = get_long_short_vector(m, k)
    for i in range(n):
        for j in range(m):
            mask_matrix[i][j] = f_vector[(j-i)%m]
    return mask_matrix

def get_long_short_vector(m=int, k=int):
    f_vector = np.zeros(m)
    if k == 1:
        f_vector[0] = 1
    else:
        for i in range(k/2):
            f_vector[i] = 1
        for i in range((k+1)/2):
            f_vector[k/2 + i * (m-k/2) / ((k+1)/2)] = 1
    return f_vector

def random_graph_list_of_p(p,layer_sizes):
    if(len(p) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of p is not the same as the length of layers")
    layerMasks = []
    for i in range(0,len(layer_sizes)-1):
        inNum = layer_sizes[i]
        outNum = layer_sizes[i+1]
        layerMasks.append((np.random.random((inNum,outNum)) <= p[i]).astype(int))
    return(layerMasks)

def fibonacci(k,layer_sizes):
    if(len(k) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of k is not the same as the length of layers")
    layerMasks = []
    for i in range(0,len(layer_sizes)-1):
        inNum = layer_sizes[i]
        outNum = layer_sizes[i+1]
        layerMasks.append(fibonacci_sparse_matrix(inNum,outNum,k[i]))
    return(layerMasks)

def long_short(k,layer_sizes):
    if(len(k) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of k is not the same as the length of layers")
    layerMasks = []
    for i in range(0,len(layer_sizes)-1):
        inNum = layer_sizes[i]
        outNum = layer_sizes[i+1]
        layerMasks.append(long_short_sparse_matrix(inNum,outNum,k[i]))
    return(layerMasks)

def random_expander(d,layer_sizes):
    if(len(k) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of d is not the same as the length of layers")
    layerMasks = []
    for i in range(0,len(layer_sizes)-1):
        inNum = layer_sizes[i]
        outNum = layer_sizes[i+1]
        random_matrix = np.random.random((inNum,outNum))
        partitions = np.partition(random_matrix,-d)[:,-d][:,np.newaxis] #find the indexes of the dth largest elements
        layerMasks.append(np.greater_equal(a,partitions).astype(int))
    return(layerMasks)



# aliases
pseudorect2 = pseudo_random_rect_graph_2
pseudorect1 = pseudo_random_rect_graph_1
pseudosquare2 = pseudo_random_square_graph_2
pseudosquare1 = pseudo_random_square_graph_1
random =  random_graph_list_of_p
expander = random_expander


from keraspatal.utils.generic_utils import get_from_module

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'graph_creation', instantiate=True, kwargs=kwargs)


# if __name__=='__main__':

#     print(pseudo_random_square_graph_1(10,3))
#     #print(pseudo_random_square_graph_2(10,[1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 1, 0.5]))

#     #print(pseudo_random_rect_graph_1(10, 5, 3))
#     #print(pseudo_random_rect_graph_2(10, 5, [1, 0.5, 0.5, 0.5, 0, 0, 0, 0, 1, 0.5]))