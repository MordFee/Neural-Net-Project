import os, json
from time import gmtime, strftime

experimentName = 'Experiment_2'

def get_json_file_name(layerSizes,
						graphParameterName=None,
						graphParameterValue=None,
						time = True,
						fileType=".json"):
    # Get the name of the file
	outputFile = 'LayerSize = '
	for l in layerSizes:
		outputFile +=  str(l) + '_'
	if graphParameterValue == None:
		outputFile += '_fc' #Fully connected
	else:
		outputFile += '_'+graphParameterName+'='+str(graphParameterValue)
	if time:
		outputFile +=   str(strftime("%Y%m%d_%Hh%Mm%Ss", gmtime()))
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
seed =333 #random esed set to replicate

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
validation_split =0
show_accuracy =True
verbose = 2

JSONDict['FitNetwork'] = {"nb_epoch":nb_epoch, 
						  "batch_size":batch_size, 
						  "validation_split":validation_split, 
						  "show_accuracy":show_accuracy, 
						  "verbose":verbose}

##THE ABOVE ARE STATIC SETTINGS NOW DYNAMIC ONES GO BELOW
try:
	os.mkdir(experimentName)
except:
	print("Experiment Already Exists")
#Params to vary
p_vals = list(x/100 for x in range(0,100))
print(p_vals)
for p in p_vals:
	JSONDict['GenerateLayerMasks']['p'] = p
	outputFile = get_json_file_name(layerSizes,"p",p)
	with open(os.path.join(experimentName,outputFile), 'w') as outfile :
		json.dump(JSONDict, outfile)