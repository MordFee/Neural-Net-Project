from utils import *
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from os.path import isfile

class Patal:

    def __init__(self,layer_sizes, fileName='mnist_14x14', activation_function='sigmoid'):
        self.layer_sizes = layer_sizes
        self.fileName = fileName
        self.activation_function = activation_function

    def run(self):
        self.get_and_reshape_datasets()
        self.create_network()
        self.fit_network()
        self.save_resutls()

    def get_and_reshape_datasets(self):
        # Get the datasets
        dataPath = '../data/' + fileName + '.pkl.gz'

        self.datasets = load_data(dataPath)
        X_train, y_train = self.datasets[0]
        X_test, y_test = self.datasets[2]

        # Reshape datasets
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = y_test
        input_num = X_train.shape[1]
        output_num = self.y_train.shape[1]
        if input_num != self.layer_sizes[0]:
            raise Exception("ERROR, the first layer does not have the right amount of neurons.")
        if output_num != self.layer_sizes[-1]:
            raise Exception("ERROR, the last layer does not have the right amount of neurons.")
        self.X_train = X_train.reshape([len(X_train), self.layer_sizes[0]])
        self.X_test = X_test.reshape([len(X_test), self.layer_sizes[0]])

    def create_network(self):
        # Create the network
        model = Sequential()
        for l in range(1, len(self.layer_sizes)):
            model.add(Dense(self.layer_sizes[l-1], self.layer_sizes[l]))
            model.add(Activation(activation_function))
            model.add(Dropout(0.01))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
        self.model = model

    def fit_network(self):
        # Run the network
        self.output = self.model.fit(self.X_train, self.y_train, nb_epoch=10, batch_size=16, validation_split=0.1, show_accuracy=True, verbose=2)

        # Get the predictions
        self.y_pred = self.model.predict_classes(self.X_test, verbose=0).astype(int)
        y_test = np.squeeze(self.y_test).astype(int)
        self.finalScore = float(sum(self.y_pred==y_test))/len(y_test)
        print self.finalScore

    def save_resutls(self):
        # Get the name of the file
        outputPath = '../results/' + self.fileName
        for l in layer_sizes:
            outputPath += '_' + str(l)
        outputPath += '_fc' #Fully connected
        outputPath += '.csv'

        # test if there is not already a file with the same name
        c = 0
        while isfile(outputPath) and c < 5:
            c += 1
            outputPath = outputPath[:-4] + '_bis.csv'
            print 'ATTENTION: the file name already exists'

        # Save the output in a csv
        df = pd.DataFrame(self.output.history)
        df.to_csv(outputPath)


if __name__=='__main__':
    # Variable
    # 'mnist_14x14'
    fileName =  'mnist'
    layer_sizes = [28*28, 128, 64, 32, 10]
    activation_function = 'sigmoid'

    # Create and run the Patal
    patal = Patal(layer_sizes, fileName=fileName, activation_function=activation_function)
    patal.run()





