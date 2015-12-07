import numpy as np
import random

# Vector

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

def get_random_vector(m=int, k=int):
    rand_vector = np.arange(m)
    np.random.shuffle(rand_vector)
    for i in range(m):
        rand_vector[i] = 1 if rand_vector[i] < k else 0
    return rand_vector

def get_regular_vector(m=int, k=int):
    reg_vector = np.zeros(m)
    for i in range(k):
        reg_vector[i] = 1
    return reg_vector

# Matrix

def fibonacci_sparse_matrix(n=int, m=int, k=int):
    mask_matrix = np.zeros( (n,m))
    f_vector = get_fibonacci_vector(m, k)
    for i in range(n):
        for j in range(m):
            mask_matrix[i][j] = f_vector[(j-i)%m]
    return mask_matrix

def long_short_sparse_matrix(n=int, m=int, k=int):
    mask_matrix = np.zeros( (n,m))
    f_vector = get_long_short_vector(m, k)
    for i in range(n):
        for j in range(m):
            mask_matrix[i][j] = f_vector[(j-i)%m]
    return mask_matrix

def regular_matrix(n=int, m=int, k=int):
    mask_matrix = np.zeros( (n,m))
    reg_vector = get_regular_vector(m, k)
    for i in range(n):
        for j in range(m):
            mask_matrix[i][j] = reg_vector[(j-i)%m]
    return mask_matrix

def rand_vector_circulant_matrix(n=int, m=int, k=int):
    mask_matrix = np.zeros( (n,m))
    f_vector = get_random_vector(m, k)
    for i in range(n):
        for j in range(m):
            mask_matrix[i][j] = f_vector[(j-i)%m]
    return mask_matrix


# Graph

def random_graph(degrees=[], layer_sizes=[]):
    if(len(degrees) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of degree is not the same as the length of layers")
    layer_masks = []
    for i in range(0,len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        layer_masks.append((np.random.random((input_size,output_size)) <= degrees[i]/float(output_size)).astype(int))
    return(layer_masks)

def fibonacci_graph(degrees=[], layer_sizes=[]):
    if(len(degrees) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of degrees is not the same as the length of layers")
    layer_masks = []
    for i in range(0,len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        layer_masks.append(fibonacci_sparse_matrix(input_size,output_size,degrees[i]))
    return(layer_masks)

def long_short_graph(degrees=[], layer_sizes=[]):
    if(len(degrees) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of degrees is not the same as the length of layers")
    layer_masks = []
    for i in range(0,len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        layer_masks.append(long_short_sparse_matrix(input_size,output_size,degrees[i]))
    return(layer_masks)

def regular_graph(degrees=[], layer_sizes=[]):
    if(len(degrees) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of degrees is not the same as the length of layers")
    layer_masks = []
    for i in range(0,len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        layer_masks.append(regular_matrix(input_size,output_size,degrees[i]))
    return(layer_masks)

def random_vector_cirgulant_graph(degrees=[], layer_sizes=[]):
    if(len(degrees) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of degrees is not the same as the length of layers")
    layer_masks = []
    for i in range(0,len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        layer_masks.append(rand_vector_circulant_matrix(input_size,output_size,degrees[i]))
    return(layer_masks)

def random_expander_graph(degrees=[], layer_sizes=[]):
    if(len(degrees) != len(layer_sizes)-1):
        raise Exception("ERROR, the length of degrees is not the same as the length of layers")
    layer_masks = []
    for i in range(0,len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        random_matrix = np.random.random((input_size,output_size))
        partitions = np.partition(random_matrix,-degrees[i])[:,-degrees[i]][:,np.newaxis] #find the indexes of the degree th largest elements
        layer_masks.append(np.greater_equal(random_matrix,partitions).astype(int))
    return(layer_masks)

def regular_expander_graph(degrees=[], layer_sizes=[]):
    #TODO 
    return



from keraspatal.utils.generic_utils import get_from_module

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'graph_creation', instantiate=True, kwargs=kwargs)
