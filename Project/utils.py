
import gzip
import cPickle
import theano
import numpy as np
from theano import tensor as T
from scipy.misc import imresize

def load_data(dataPath, use_gpu=False):
    """Download data
    train_set, valid_set, test_set format: tuple(input, target)
    input is an numpy.ndarray of 2 dimensions (a matrix)
    witch row's correspond to an example. target is a
    numpy.ndarray of 1 dimensions (vector)) that have the same length as
    the number of rows in the input. It should give the target
    target to the example with the same index in the input.
    """

    # Download data
    f = gzip.open(dataPath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # Preformat data if the program uses GPU

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if use_gpu:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

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