from Layer import *
from utils import *
from numpy.linalg import norm as norm
from numpy.random import normal as normal
from numpy import sum as sum

class MultiLayer:
    """
    nummberOfNeurons = [nb neurons in input, nb neurons in hidden layer 1, ... , nb neuron in output]
    activationFunction = the activation function used in the network
    firstLayerNumber or l refers to the layer number. 0 is the input layer, L is the output layer
    """

    def __init__(self, nummberOfNeurons, activationFunction, bFillFunction=normal(0,1), wFillFunction=normal(0,1)):
        self.numberOfLayer = len(nummberOfNeurons)
        self.numberOfNeurons = nummberOfNeurons
        self.totalNumberOfNeuron = sum(nummberOfNeurons)
        self.activationFunction = activationFunction
        self.layers = []
        for l in xrange(self.numberOfLayer - 1):
            self.layers.append(Layer(self.numberOfNeurons[l], self.numberOfNeurons[l+1], activationFunction, bFillFunction, wFillFunction))

    def run(self, networkInput):
        """
        Run the whole network.
        Takes networkInput of the size of the first layer as input vector.
        Returns networkOutput of the size of the last layer as output vector
        """
        currentInput = networkInput
        currentOutput = []
        for l in xrange(self.numberOfLayer - 1):
            currentOutput = self.layers[l].run_layer(currentInput)
            currentInput = currentOutput
        networkOutput = currentOutput
        return networkOutput

    def run_with_changed_w(self, networkInput, firstLayerNumber, firstNeuronNumber, secondNeuronNumber, addedWeight):
        """
        Run the network with layer[l].w[secondNeuronNumber][firstNeuronNumber] += addedWeight
        """
        currentInput = networkInput
        currentOutput = []
        for l in xrange(self.numberOfLayer - 1):
            if l != firstLayerNumber:
                currentOutput = self.layers[l].run_layer(currentInput)
                currentInput = currentOutput
            else:
                currentOutput = self.layers[l].run_layer_with_changed_w(currentInput, firstNeuronNumber, secondNeuronNumber, addedWeight)
                currentInput = currentOutput
        networkOutput = currentOutput
        return networkOutput

    def run_with_changed_b(self, networkInput, firstLayerNumber, secondNeuronNumber, addedWeight):
        """
        Run the network with layer[l].b[secondNeuronNumber] += addedWeight
        """
        currentInput = networkInput
        currentOutput = []
        for l in xrange(self.numberOfLayer - 1):
            if l != firstLayerNumber:
                currentOutput = self.layers[l].run_layer(currentInput)
                currentInput = currentOutput
            else:
                currentOutput = self.layers[l].run_layer_with_changed_b(currentInput, secondNeuronNumber, addedWeight)
                currentInput = currentOutput
        networkOutput = currentOutput
        return networkOutput

    def learn_by_backpropagation(self, networkInput, wantedOutput, eta, delta):
        """
        eta is the parameter for the descent.
            If eta is too small, the learning rate will be small.
            if eta is too big, we might nerver find the minimum.
        delta is the closeness between the networkOutput and wantedOutput.
            If delta is too small, we will be overfiting.
            If delta is too big, the network won't be efficient.
        """
        currentOutput = self.run(networkInput)
        diff = norm( map(substraction, currentOutput, wantedOutput))
        cpt = 0
        while diff > delta and cpt < 100:
            cpt += 1
            for l in xrange(self.numberOfLayer - 1, -1, -1): # For back propagation
                for j in xrange(len(self.layers[l+1])):
                    for i in xrange(len(self.layers[l])):
                        currentOutput = self.run(networkInput)
                        currentOutput2 = self.run_with_changed_w(networkInput, l, i, j, eta)
                        self.layers[l].w[j][i] -=  (currentOutput2 - currentOutput )
                    currentOutput = self.run(networkInput)
                    currentOutput2 = self.run_with_changed_b(networkInput, l, j, eta)
                    self.layers[l].b[j] -=  (currentOutput2 - currentOutput )
            currentOutput = self.run(networkInput)
            diff = norm( map(substraction, currentOutput, wantedOutput))
        return

if __name__ == "__main__":
    set_x = quick_load()
    images, labels = set_x[0][:1], set_x[1][:1]

    print images, labels
    myInput = []
    #myNeurons = [164,80,10]
    #myNetwork = MultiLayer(myNeurons, sigmoid)

