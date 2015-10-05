import networkx as nx
import keras
import numpy as np
import matplotlib,re


def keras_to_graph(model,layerNames,threshhold=0): #can change to list of threshholds or some methods or something
    '''
    This function converts a keras neural network to a NetworkX graph
    Inputs:
    -model, the keras model
    -layerNames, the list of layer names
    -threshhold(optional) - the absolute value above which the edge should be drawn
    '''
    graph = nx.DiGraph() #Use directed graph for now
    for ix,layerName in enumerate(layerNames):
        if ix >0:
            graph = graph_add_layer(graph,layerNames[ix-1],layerNames[ix],model.get_weights()[(ix-1)*2],threshhold)
    return(graph)
def separate_layers(listOfNodes,layerNames,sep='_'):
    """
    This function identifies the layers of a neural network based on the names of the layers
    Inputs:
    -a list of strings corresponding to nodes in a graph
    -a list of layer names
    Returns:
    -a list of lists with the names of nodes of each layer in a separate list
    """
    layers =[]
    for layer in layerNames:
        temp = []
        for node in listOfNodes:
            if re.search(layer+sep,node):
                temp.append(node)
        layers.append(temp)
    return layers
def plot_forward_neural_net(graph,layerNames,weighted=False, scaling= lambda x:x):
    '''
    This function returns a matplotlib object of a bipartite neural net
    It takes as input:
    -a NetworkX Graph
    -a list of layer names (could be deprecated with a regex but w/e)
    '''
    
    ##TODO: Make it return the Matplotlib object or figure isntead of automatically drawing
    
    listOfLayers = separateLayers(graph.nodes(),layerNames)
    positions = dict()
    for ix,layer in enumerate(listOfLayers):
        positions.update((n, (ix, i*(1-2*(i%2)))) for i, n in enumerate(layer))
    if weighted:
        nx.draw_networkx_nodes(graph,pos=positions) #draw the nodes
        #Loop through each edge 
        for (fromNode,toNode,weight) in graph.edges(data=True):
            nx.draw_networkx_edges(graph,positions,edgelist=[(fromNode,toNode)],width=scaling(weight['weight']))
    else:
        nx.draw(graph, pos=positions)

def graph_add_layer(graph, fromLayer,toLayer,connectionMatrix,threshhold=0,sep='_'):
    '''
    This method grows a graph according to non-zero connection matrix for to layers in a neural net
    This method takes as inputs:
    -a NetworkX object, graph
    -the name of the fromLayer and toLayer
    -a connection matrix of dimension fromLayer x toLayer
    and returns:
    -the grown graph
    '''
    fromNodeNames = list(map(lambda x: str(fromLayer) + sep + str(x),range(connectionMatrix.shape[0]))) #generate the names for the fromLayer
    toNodeNames = list(map(lambda x: str(toLayer) + sep + str(x),range(connectionMatrix.shape[1]))) #generate the names for the toLayer
    
    #add the nodes to the graph
    graph.add_nodes_from(fromNodeNames + toNodeNames)
    
    booleanMatrix = np.abs(connectionMatrix) > threshhold 
    
    #generate the edges
    edges = []
    for fromIX,fromNode in enumerate(fromNodeNames):
        for toIX,toNode in enumerate(toNodeNames):
            if(booleanMatrix[fromIX][toIX]):
                edges.append((fromNode,toNode, connectionMatrix[fromIX][toIX]))
    graph.add_weighted_edges_from(edges)
    
    return graph