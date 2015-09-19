

class Layer:
    """
    Linker is the links between layer l (firstLayer) and layer l+1 (secondLayer)
    It contains the weights:
    - w[j][i] between neuron i of layer l and neuron j of layer l+1
    - b[i] of neuron i in layer l+1
    """
    def __init__(self, firstLayerSize, secondLayerSize, activationFunction, bLinksFillFunction, wLinksFillFunction):
        self.firstLayerSize = firstLayerSize
        self.secondLayerSize = secondLayerSize
        self.activationFunction = activationFunction
        self.bLinks = []
        self.wijLinks = []
        for i in xrange(secondLayerSize):
            self.bLinks.append(bLinksFillFunction)
            self.wijLinks.append([])
            for j in xrange(firstLayerSize):
                self.wijLinks[j].append(wLinksFillFunction)

    def run_layer(self, layerInput):
        layerOutput = []
        for i in xrange(self.secondLayerSize):
            layerOutput.append(0)
            for j in xrange(self.firstLayerSize):
                layerOutput[-1] += self.wijLinks[j][i] * layerInput[j]
            layerOutput[-1] = self.activationFunction(layerOutput[-1]) + self.bLinks[i]
        return layerOutput
