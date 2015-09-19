from numpy import dot as dot

class Layer:
    """
    The Layer object represents the links between layer l (firstLayer) and layer l+1 (secondLayer)
    firstNeuronNumber or i always refers to firstLayer (l)
    secondNeuronNumber or j always refers to secondLayer (l+1)
    It contains the weights:
    - w[j][i] between neuron i of layer l and neuron j of layer l+1
    - b[j] of neuron j in layer l+1
    """
    def __init__(self, firstLayerSize, secondLayerSize, activationFunction, bFillFunction, wFillFunction):
        self.firstLayerSize = firstLayerSize
        self.secondLayerSize = secondLayerSize
        self.activationFunction = activationFunction
        self.b = []
        self.w = []
        for j in xrange(secondLayerSize):
            self.b.append(bFillFunction)
            self.w.append([])
            for i in xrange(firstLayerSize):
                self.w[j].append(wFillFunction)

    def run_layer(self, layerInput):
        """
        Run the layer alone.
        Takes layerInput of the size of the first layer as input vector.
        Returns layerOutput of the size of the second layer as output vector.
        """
        layerOutput = []
        for j in xrange(self.secondLayerSize):
            layerOutput.append( self.activationFunction( dot( self.w[j], layerInput) + self.b[j] ))
        return layerOutput

    def run_layer_with_changed_w(self, layerInput, firstNeuronNumber, secondNeuronNumber, addedWeight):
        """
        Run the layer with w[secondNeuronNumber][firstNeuronNumber] += addedWeight.
        """
        self.w[secondNeuronNumber][firstNeuronNumber] += addedWeight  # Change the weight
        layerOutput = self.run_layer(layerInput)                      # Run the layer
        self.w[secondNeuronNumber][firstNeuronNumber] -= addedWeight  # Put the weight back
        return layerOutput

    def run_layer_with_changed_b(self, layerInput, secondNeuronNumber, addedWeight):
        """
        Run the layer with b[secondNeuronNumber] += addedWeight.
        """
        self.b[secondNeuronNumber] += addedWeight  # Change the weight
        layerOutput = self.run_layer(layerInput)   # Run the layer
        self.b[secondNeuronNumber] -= addedWeight  # Put the weight back
        return layerOutput