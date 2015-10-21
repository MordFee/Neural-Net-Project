import __future__ 
from utils import *
from graph_utils import *
import graph_creation
import numpy as np
import pandas as pd
from keraspatal.models import Sequential
from keraspatal.layers.core import Dense, Dropout, Activation
from keraspatal.optimizers import SGD
from keraspatal.utils import np_utils
import networkx as nx
import warnings, sys, glob, os, json,random
from time import gmtime, strftime
from sklearn import metrics
from functools import reduce
import keras.datasets.mnist as mnist

class Patal:
    def __init__(self,dataFileName='mnist_14x14',*args):
        '''
        Initialize the Patal class with the data set to be used, so that multiple experiments can be done with the same data:
            -dataFileName: the name of the data set to be read in, defaults to mnist_14x14
        '''
        self.dataFileName = dataFileName
        self.get_and_reshape_datasets(dataFileName)


    def get_and_reshape_datasets(self,dataFileName):
        # Get the datasets -- Deprecated because keras comes with the dataset
        # dataPath = 'Data/' + dataFileName + '.pkl.gz'

        # self.datasets = load_data(dataPath)
        # XTrain, yTrain = self.datasets[0]
        # XTest, yTest = self.datasets[1]

        (XTrain, yTrain), (XTest, yTest) = mnist.load_data()
        # Reshape datasets
        yTrain = np_utils.to_categorical(yTrain)
        self.inputNum = XTrain.size/len(XTrain)
        self.outputNum = yTrain.size/len(yTrain)

        self.XTrain = XTrain.reshape([len(XTrain), self.inputNum])
        self.XTest = XTest.reshape([len(XTest), self.inputNum])
        self.yTrain = yTrain
        self.yTest = yTest

    def generate_layer_masks(self,graphGeneratorAlias,graphGeneratorParams,seed=333):
        random.seed(seed)
        layerMasks = graph_creation.get(graphGeneratorAlias,graphGeneratorParams)
        return(layerMasks)

    def create_network(self, 
                       layerSizes, 
                       layerNames=None, 
                       layerMasks=None,
                       dropout=0,  
                       activationFunction='sigmoid',
                       loss='mse', 
                       lr=0.01, 
                       decay=1e-6, 
                       momentum=0.9, 
                       nesterov=True,
                       optimizer = 'sgd',
                       init='glorot_uniform'):

        self.layerSizes = layerSizes
        if self.inputNum != self.layerSizes[0]:
            raise Exception("ERROR, the first layer does not have the right amount of neurons.")
        if self.outputNum != self.layerSizes[-1]:
            raise Exception("ERROR, the last layer does not have the right amount of neurons.")
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

        if layerMasks != None: #Need to generate it based on the graph generations!!
            self.layerMasks = layerMasks
        else: #intialize all masks to None type, makes conditionals later one asier. Handled internally in keraspatal
            self.layerMasks = [None for i in layerSizes]

        # Create the network
        model = Sequential()
        for l in range(1, len(self.layerSizes)):
            #Might have to initialize the weights with no connection to zero... there will be an initial update using the connections though shouldn't make a huge difference
            model.add(Dense(
                        input_dim = self.layerSizes[l-1], 
                        output_dim = self.layerSizes[l],
                        activation=activationFunction,
                        init = init
                        ),
                    self.layerMasks[l-1]) #add the desired mask, works in keraspatal
            if dropout > 0:
                model.add(Dropout(dropout))
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        model.compile(loss=loss, optimizer=sgd)
        self.model = model

    def fit_network(self,
                     nb_epoch=20, 
                     batch_size=16, 
                     validation_split=0, 
                     show_accuracy=True, 
                     verbose='patal',
					 output_filepath='output_filepath.txt'
                     ):
        # Run the network
        # from keraspatal.callbacks import ModelCheckpoint 
        # checkpoint = ModelCheckpoint(filepath="tmp/weights.hdf5", verbose=1)
        # callbacks=[checkpoint]
        myfile = open(output_filepath, 'w')
        myfile.write('Timing (s), Loss, Accuracy\n')
        myfile.close()
        self.output = self.model.fit(self.XTrain, 
                                    self.yTrain, 
                                    nb_epoch=nb_epoch, 
                                    batch_size=batch_size, 
                                    validation_split=validation_split,
                                    show_accuracy=show_accuracy, 
                                    verbose=verbose, 
                                    # callbacks=callbacks,
									output_file=output_filepath
									)

        self.yPred = self.model.predict_classes(self.XTest, verbose=0).astype(int)
        yTest = np.squeeze(self.yTest).astype(int)
        self.finalScore = float(sum(self.yPred==yTest)) / len(yTest)
        myfile = open(output_filepath, 'a')
        myfile.write(str(self.finalScore) + '\n')
        myfile.write(str(metrics.confusion_matrix(yTest, self.yPred)) + '\n')
        myfile.close()

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
        try:
            self.model.load_weights(filePath)
            return(True)
        except:
            print('Was unable to save the model to %s' % filePath)
            return(False)            
    # def save_configs(self,filePath):
    #     #TODO: save the configurations of this model to a JSON
    # def load_configs(self, filepath):
    #     #TODO: create a patal class from JSON config 

    def get_file_name(self,file_type=".csv"):
        # Get the name of the file
        outputPath = '../results/' + self.dataFileName + str(strftime("%Y%m%d_%Hh%Mm%Ss", gmtime()))
        for l in layerSizes:
            outputPath += '_' + str(l)
        outputPath += '_fc' #Fully connected
        outputPath += fileType

    def save_results(self):
        # Save the output in a csv
        outputPath = self.get_file_name()
        df = pd.DataFrame(self.output.history)
        df.to_csv(outputPath)

    
if __name__=='__main__':

    patal=Patal()

    for folderIX in range(1,len(sys.argv)): #for each folde rin the input
        folder = sys.argv[folderIX] #get that folders path
        if not os.path.isdir(folder): #make sure it exists
            print("The director %s does not exists" % (folder))
            continue
        #create an output folder in the  outputDir
        folderName = folder.split('/')[-1] 
        outputDir = 'Results/'+ folderName
        try:
            os.mkdir(outputDir)
        except:
            print("Folder already Exists")
        outputFile = open(outputDir+folderName+'.csv', 'w')
        outputFile.write('P Value, Accuracy\n')
        #For each JSON in the Experiment in question
        for configJSONFile in glob.glob(folder+'/*.json'):
            try: #Load in the JSON
                JSONData = open(configJSONFile).read()
                JSONDict = json.loads(JSONData)
            except:
                print("Could not read in file %s" %(configJSONFile))
                continue
            #Run the model
            print("Generating Layer Masks")
            patal.generate_layer_masks(**JSONDict['GenerateLayerMasks'])
            print("Creating Network")
            patal.create_network(**JSONDict['CreateNetwork'])
            print("Training Network")
            patal.fit_network(**JSONDict['FitNetwork'])
            patal.save_model(JSONDict['FitNetwork']['output_filepath'][:-5]+'.hdf5')
            #Need to save weights
            outputFile.write(str(JSONDict['GenerateLayerMasks']['graphGeneratorParams']['p'])+','+str(self.finalScore)+'\n')
        outputFile.close()