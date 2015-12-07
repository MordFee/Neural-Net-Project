import os, json, random
from time import gmtime, strftime
from json import encoder




def get_json_file_name(layer_sizes,
                        degrees,
                        alias,
                        fileType=".json",
                        unique_id=''):
    # Get the name of the file
    outputFile = 'Layers=' + str(layer_sizes)
    outputFile += '_Degrees=' + str(degrees)
    outputFile += '_Alias=' + str(alias)
    outputFile += '_' + unique_id + fileType
    return(outputFile)


def get_JSON_dict(degrees,
                  layer_sizes,
                  graphGeneratorAlias,
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

    graphGeneratorParams = {'degrees' : degrees,
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
    #unique_id = str(strftime("%Y%m%d_%Hh%Mm%Ss", gmtime()))

    hidden_layer = 100
    graph_alias = []
    graph_alias.append('random_graph')
    graph_alias.append('fibonacci_graph')
    graph_alias.append('long_short_graph')
    graph_alias.append('regular_graph')
    graph_alias.append('random_vector_cirgulant_graph')
    graph_alias.append('random_expander_graph')
    #graph_alias = 'regular_expander_graph'
    degree_hidden = [2,6,10]
    for current_hidden_degree in degree_hidden:
        for alias in graph_alias:
            unique_id = 'H=' + str(hidden_layer)
            unique_id += '_D=' + str(current_hidden_degree)
            unique_id += '_A=' + str(alias)

            experiment_name = 'Experiment_'  + unique_id
            os.mkdir(experiment_name)
            JSON_dict = {}

            nb_epoch = 50
            batch_size = 32

            layer_sizes = [784, hidden_layer, 10]
            degree_list1 = [i for i in range(1, 30)]
            degree_list2 = [10*i for i in range(3, 11)]
            degree_list = degree_list1 + degree_list2
            print degree_list
            for degree in degree_list:


                degrees = [degree, current_hidden_degree]

                JSON_dict = get_JSON_dict(degrees, layer_sizes, alias, nb_epoch=nb_epoch, batch_size=batch_size)



                #Set seed as well
                JSON_dict['GenerateLayerMasks']['seed'] = random.randint(0,1000)
                #and the output file name
                output_filepath = get_json_file_name(layer_sizes, degrees, alias, fileType='.csv')
                JSON_dict['FitNetwork']['output_filepath'] = output_filepath

                output_file_JSON = get_json_file_name(layer_sizes, degrees, alias)
                f = open(os.path.join(experiment_name, output_file_JSON), 'w')
                json.dump(JSON_dict, f)
                f.close()

            # Save this JSON creator file
            saved_make_exp = open(os.path.join(experiment_name, 'make_experiment.py'), 'w')
            make_exp_file = open('make_experiment.py', 'r')
            saved_make_exp.write(make_exp_file.read())
            make_exp_file.close()
            saved_make_exp.close()