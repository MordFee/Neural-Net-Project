import os, json, random
from time import gmtime, strftime
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

experimentName = 'Experiment_lists'

def get_json_file_name(layerSizes,
						graphParameterName=None,
						graphParameterValue=None,
						time = True,
						fileType=".json"):
    # Get the name of the file
	outputFile = 'LayerSize='
	for l in layerSizes:
		outputFile +=  str(l) + '_'
	if graphParameterValue == None:
		outputFile += '_fc' #Fully connected
	else:
		outputFile += graphParameterName + '=' + str(graphParameterValue)
	outputFile += fileType
	return(outputFile)

JSONDict = {}
##General Topology
layerSizes = [784,300,150,10]

###-----------------Generate Layer Masks
'''
Alias for which function to generate the connections for the layer
Current Options are:
-random Inputs: p
-psuedorect1
-psuedorect2
-psuedosquare1
-psuedosquare2
'''

graphGeneratorAlias = 'random' 
graphGeneratorParams = {'p' : .6,
						"layerSizes" : layerSizes} #parameters to be passed to the graph creating function
seed = random.randint(0,1000) #random esed set to replicate

JSONDict['GenerateLayerMasks'] ={"graphGeneratorAlias" : graphGeneratorAlias,
								"graphGeneratorParams" : graphGeneratorParams,
								"seed" : seed}

###-----------------Create Network

#Refer to the keras documentation for layer creation
layerSizes = layerSizes
layerNames = ["input","L1","L2","output"]
dropout = 0
activationFunction= "sigmoid"
loss="mse"
lr=0.01
decay=1e-6
momentum=0.9
nesterov=True
optimizer = "sgd"
init="glorot_uniform"

JSONDict["CreateNetwork"] = {"layerSizes" : layerSizes,
							"layerNames" : layerNames,
							"dropout" : dropout,
							"activationFunction": activationFunction,
							"loss": loss, 
							"lr":lr, 
							"decay":decay, 
							"momentum":momentum, 
							"nesterov":nesterov,
							"optimizer" : optimizer,
							"init": init}

####--------------------------------Fit Network

#Refer to Keras documentation
nb_epoch = 20
batch_size = 16 
validation_split = 0
show_accuracy =True
verbose = 'patal'
output_filepath= 'output_filepath.txt'

JSONDict['FitNetwork'] = {"nb_epoch":nb_epoch, 
						  "batch_size":batch_size, 
						  "validation_split":validation_split, 
						  "show_accuracy":show_accuracy, 
						  "verbose":verbose,
						  "output_filepath":output_filepath}

##THE ABOVE ARE STATIC SETTINGS NOW DYNAMIC ONES GO BELOW
try:
	os.mkdir(experimentName)
except:
	print("Experiment Already Exists")
#Params to vary
#p_vals = list(float(x)/100. for x in range(1,31))
numberOfvalues = 100
rangeOfValues = range(0,numberOfvalues)
p_val_input = [1 for i in rangeOfValues]
p_val_layer_1 = [.5 for i in rangeOfValues]
p_val_layer_2 = [float(x)/100. for x in rangeOfValues]
p_val_output = [1 for i in rangeOfValues]

p_vals = [list(i) for i in zip(p_val_input,p_val_layer_1,p_val_layer_2,p_val_output)]
print p_vals
for p in p_vals:
	JSONDict['GenerateLayerMasks']['graphGeneratorParams']['p'] = p
	#Set seed as well
	JSONDict['GenerateLayerMasks']['seed'] = random.randint(0,1000)
	#and the output file name
	output_filepath = get_json_file_name(layerSizes, "p", p, '.csv')
	JSONDict['FitNetwork']['output_filepath'] = output_filepath

	outputFileJSON = get_json_file_name(layerSizes, "p", p)
	with open(os.path.join(experimentName, outputFileJSON), 'w') as outfile :
		json.dump(JSONDict, outfile)