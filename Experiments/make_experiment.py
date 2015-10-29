import os, json, random
from time import gmtime, strftime
from json import encoder




def get_json_file_name(layer_sizes,
                        graphParameterName=None,
                        graphParameterValue=None,
                        fileType=".json",
                        unique_id=''):
    # Get the name of the file
    outputFile = 'LayerSize='
    for l in layer_sizes:
        outputFile +=  str(l) + '_'
    if graphParameterValue == None:
        outputFile += '_fc' #Fully connected
    else:
        outputFile += graphParameterName + '=' + str(graphParameterValue)
    outputFile += '_' + unique_id + fileType
    return(outputFile)


def get_JSON_dict(layer_sizes,
                  nb_epoch=20,
                  batch_size=16):
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
                            "layer_sizes" : layer_sizes} #parameters to be passed to the graph creating function
    seed = random.randint(0,1000) #random esed set to replicate

    JSON_dict['GenerateLayerMasks'] ={"graphGeneratorAlias" : graphGeneratorAlias,
                                    "graphGeneratorParams" : graphGeneratorParams,
                                    "seed" : seed}

    ###-----------------Create Network

    #Refer to the keras documentation for layer creation
    layer_sizes = layer_sizes
    dropout = 0
    activation_function= "sigmoid"
    loss="mse"
    lr=0.01
    decay=1e-6
    momentum=0.9
    nesterov=True
    optimizer = "sgd"
    init="glorot_uniform"

    JSON_dict["CreateNetwork"] = {"layer_sizes" : layer_sizes,
                                "dropout" : dropout,
                                "activation_function": activation_function,
                                "loss": loss,
                                "lr":lr,
                                "decay":decay,
                                "momentum":momentum,
                                "nesterov":nesterov,
                                "optimizer" : optimizer,
                                "init": init}

    ####--------------------------------Fit Network

    #Refer to Keras documentation
    validation_split = 0
    show_accuracy =True
    verbose = 'patal'
    output_filepath= 'output_filepath.txt'

    JSON_dict['FitNetwork'] = {"nb_epoch":nb_epoch,
                              "batch_size":batch_size,
                              "validation_split":validation_split,
                              "show_accuracy":show_accuracy,
                              "verbose":verbose,
                              "output_filepath":output_filepath}
    return(JSON_dict)

##THE ABOVE ARE STATIC SETTINGS NOW DYNAMIC ONES GO BELOW

if __name__ == "__main__":
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    unique_id = str(strftime("%Y%m%d_%Hh%Mm%Ss", gmtime()))
    experiment_name = 'Experiment_'  + unique_id
    os.mkdir(experiment_name)
    JSON_dict = {}
    nb_epoch = 20
    batch_size = 16

    number_of_values = 90
    initial_layer_size = 100
    initial_proba = 1
    for k in range(number_of_values+1):

        ##General Topology
        moving_layer_size = initial_layer_size + 10.*k
        layer_sizes = [784, 300, moving_layer_size, 10]

        JSON_dict = get_JSON_dict(layer_sizes, nb_epoch=nb_epoch, batch_size=batch_size)

        moving_proba = initial_layer_size / moving_layer_size * initial_proba
        moving_probas = [1, 1, moving_proba, 1]
        p = moving_probas

        JSON_dict['GenerateLayerMasks']['graphGeneratorParams']['p'] = p
        #Set seed as well
        JSON_dict['GenerateLayerMasks']['seed'] = random.randint(0,1000)
        #and the output file name
        output_filepath = get_json_file_name(layer_sizes, "p", p, fileType='.csv', unique_id=unique_id)
        JSON_dict['FitNetwork']['output_filepath'] = output_filepath

        output_file_JSON = get_json_file_name(layer_sizes, "p", p, unique_id=unique_id)
        file = open(os.path.join(experiment_name, output_file_JSON), 'w')
        json.dump(JSON_dict, file)
        file.close()

    # Save this JSON creator file
    saved_make_exp = open(os.path.join(experiment_name, 'make_experiment.py'), 'w')
    make_exp_file = open('make_experiment.py', 'r')
    saved_make_exp.write(make_exp_file.read())
    make_exp_file.close()
    saved_make_exp.close()