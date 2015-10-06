
import gzip
import cPickle
import theano
import numpy as np
from theano import tensor as T
from scipy.misc import imresize

def load_data(dataPath, use_gpu=False):
    """Download data
    trainSet, validSet, testSet format: tuple(input, target)
    input is an numpy.ndarray of 2 dimensions (a matrix)
    witch row's correspond to an example. target is a
    numpy.ndarray of 1 dimensions (vector)) that have the same length as
    the number of rows in the input. It should give the target
    target to the example with the same index in the input.
    """

    # Download data
    f = gzip.open(dataPath, 'rb')
    trainSet, validSet, testSet = cPickle.load(f)
    f.close()


    XTestSet, yTestSet = testSet
    XValidSet, yValidSet = validSet
    XTrainSet, yTrainSet = trainSet

    return [(np.append(XTrainSet, XValidSet, axis=0), np.append(yTrainSet, yValidSet, axis=0)), (XTestSet, yTestSet)]

def resize(data):
    # Resizes a 28x28 image to 14x14
    numRows = data.shape[0]
    newData = np.zeros((numRows, 196))

    for i in xrange(numRows):
        a = np.reshape(data[i,:], (28,28))
        newImage = imresize(a,(14,14))
        newData[i,:] = np.reshape(newImage, (1,196))
    print("resized")
    return newData