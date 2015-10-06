from utils import *
from graph_utils import *
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import networkx as nx
import warnings
from time import gmtime, strftime
from sklearn import metrics

class Patal:

    def __init__(self, layerSizes, layerNames=None, dataFileName='mnist_14x14', activationFunction='sigmoid', graph=None):
        self.layerSizes = layerSizes
        self.dataFileName = dataFileName
        ##Assign layer names
        if layerNames is not None:
            if len(layerNames) != len(layerSizes):
                print('Warning: there aren\'t as many layerNames as layers, reassigning names')
                self.layerNames = ['input'] + ['L'+str(x) for x in range(1,len(layerSizes)-1)] + ['output']
            else:
                self.layerNames = layerNames
        else:
            self.layerNames = ['input'] + ['L'+str(x) for x in range(1,len(layerSizes)-1)] + ['output']
        self.activationFunction = activationFunction
        self.graph = graph

    def run(self):
        self.get_and_reshape_datasets()
        self.create_network()
        self.fit_network()
        self.save_results()

    def get_and_reshape_datasets(self):
        # Get the datasets
        dataPath = '../data/' + dataFileName + '.pkl.gz'

        self.datasets = load_data(dataPath)
        XTrain, yTrain = self.datasets[0]
        XTest, yTest = self.datasets[1]

        # Reshape datasets
        self.yTrain = np_utils.to_categorical(yTrain)
        self.yTest = yTest
        input_num = XTrain.shape[1]
        output_num = self.yTrain.shape[1]
        if input_num != self.layerSizes[0]:
            raise Exception("ERROR, the first layer does not have the right amount of neurons.")
        if output_num != self.layerSizes[-1]:
            raise Exception("ERROR, the last layer does not have the right amount of neurons.")
        self.XTrain = XTrain.reshape([len(XTrain), self.layerSizes[0]])
        self.XTest = XTest.reshape([len(XTest), self.layerSizes[0]])

    def create_network(self, dropout=0, loss='mse', lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
        # Create the network
        model = Sequential()
        for l in range(1, len(self.layerSizes)):
            model.add(Dense(self.layerSizes[l-1], self.layerSizes[l]))
            model.add(Activation(activationFunction))
            model.add(Dropout(dropout))
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        model.compile(loss=loss, optimizer=sgd)
        self.model = model

    def fit_network(self, nb_epoch=20, batch_size=16, validation_split=0, show_accuracy=True, verbose=2):
        # Run the network
        self.output = self.model.fit(self.XTrain, self.yTrain, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=validation_split, show_accuracy=show_accuracy, verbose=verbose)

        # Get the predictions
        self.yPred = self.model.predict_classes(self.XTest, verbose=0).astype(int)
        yTest = np.squeeze(self.yTest).astype(int)
        self.finalScore = float(sum(self.yPred==yTest)) / len(yTest)
        print(self.finalScore)
        print(metrics.confusion_matrix(yTest, self.yPred))

    def generate_graph(self, threshold=0):
        # This function generates a NetworkX graph based on the model setup
        self.graph = keras_to_graph(self.model, self.layerNames, threshold)

    def plot_graph(self, weighted=False, scaling=lambda x:x):
        # This function plots the current state of the feed forward neural net
        if self.graph == nx.classes.digraph.DiGraph:
            plot_forward_neural_net(self.graph ,self.layerNames, weighted=weighted, scaling=scaling)
        else:
            warnings.warn("Graph has not been generated correctly; cannot plot!", RuntimeWarning)

    def graph_metric(self, metric=nx.algorithms.node_connectivity, layers=True):
        '''
        This function generates a dataframe of metric pertaining to each layer's connections, and the overall network as a whole
        Returns dataframe
        '''
        cols = ["Metric Name "] + ['fullModel']
        metrics = [metric.__name__, metric(self.graph)]
        if layers:
            cols = cols + ['%s to %s' % t for t in zip(self.layerNames[:-1], self.layerNames[1:])]
            metrics = [metric(self.graph.subgraph(layer)) for layer in separate_layers(self.graph)]
        return pd.DataFrame(metrics, columns=cols)

    def save_model(self, filePath):
        self.model.save_weights(filePath)

    def load_model(self, filePath):
        self.model.load_weights(filePath)

    def get_file_name(self):
        # Get the name of the file
        outputPath = '../results/' + self.dataFileName + str(strftime("%Y%m%d_%Hh%Mm%Ss", gmtime()))
        for l in layerSizes:
            outputPath += '_' + str(l)
        outputPath += '_fc' #Fully connected
        outputPath += '.csv'

    def save_results(self):
        # Save the output in a csv
        outputPath = self.get_file_name()
        df = pd.DataFrame(self.output.history)
        df.to_csv(outputPath)


if __name__=='__main__':
    # Variable
    # 'mnist_14x14'
    dataFileName =  'mnist'
    layerSizes = [28*28, 300, 150, 10]
    activationFunction = 'sigmoid'

    # Create and run the Patal
    patal = Patal(layerSizes, dataFileName=dataFileName, activationFunction=activationFunction)
    patal.run()





