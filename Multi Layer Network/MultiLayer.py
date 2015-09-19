from Layer import *
from utils import *
from numpy.linalg import norm as norm
from numpy.random import normal as normal

class MultiLayer:
    """
    nummberOfNeurons = [nb neurons in input, nb neurons in hidden layer, ... , nb neuron in output]
    activationFunction = the activation function used in the network
    """

    def __init__(self, nummberOfNeurons, activationFunction, bLinksFillFunction=normal(0,1), wLinksFillFunction=normal(0,1)):
        self.numberOfLayer = len(nummberOfNeurons)
        self.numberOfNeuron = nummberOfNeurons
        self.activationFunction = activationFunction
        self.linkers = []
        for i in xrange(1, self.numberOfLayer):
            self.linkers.append(Layer(self.numberOfNeuron[i-1],self.numberOfNeuron[i] , bLinksFillFunction, wLinksFillFunction))

    def run(self, networkInput):
        currentOutput = []
        for i in xrange(self.numberOfLayer - 1):
            currentOutput = self.linkers[i].run_layer(networkInput)
            networkInput = currentOutput
        return currentOutput

    def run_with_changed_value(self, networkInput, linkerNumber, firstNeuronNumber, secondNeuronNumber):
        currentOutput = []
        for i in xrange(self.numberOfLayer - 1):
            pass #TODO
        return currentOutput

    def run_and_learn(self, networkInput, wantedOutput, eta, delta):
        currentOutput = self.run(networkInput)
        diff = norm( map(addition, currentOutput, wantedOutput))
        cpt = 0
        while diff > delta and cpt < 1000:
            cpt += 1
            currentOutput2 = currentOutput
            currentOutput = self.run(networkInput)
            #TODO
        return

if __name__ == "__main__":

    myInput = []
    myNeurons = [10,6,10]
    myNetwork = MultiLayer(myNeurons, sigmoid)