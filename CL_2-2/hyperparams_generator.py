"""
En este archivo se guardan las funciones utilizadas para generar hiperparametros
aleatoriamente o siguiendo algun tipo de estrategia de optimizacion
"""

from numpy.random import rand, randint, choice
import numpy as np
from dict_hash import sha256
import json



# Dimensiones de la imagen (al reves) para poder
# obtener las in_features despues del flatten
IMG_SIZE = (480, 640)
IMG_IN_CHANNELS = 1


def getRandomHyperParamsV1():
    """
    Devuelve un set de hiper parametros generado aleatoriamente dentro de
    unos umbrales aceptables.

    Devuelve una tupla:

        (hiperparametros, capas_de_la_red)

    donde hiperparametros es un dict y capas_de_la_red es una lista con las
    capas.
    """
    # Construimos los hiperparametros aleatorios
    lr = randint(1,101) * 1e-6
    ep = randint(5,41)
    tp = round((rand()*0.4)+0.5, 2)
    hyperparams = {
            "learning_rate": lr, # Float en el rango [1,100] e-6
            "batch_size": 1,
            "epochs": ep, # Int en el rango [5,40]
            "training_percent": tp, # Float en el rango [0.5, 0.9]
            "model_name": None # mas tarde se le pone el HasCode de los hyperparametros
    }

    net_layers = []

    # Creacion aleatoria de capas de la red
    # Debido a que la creacion del set convolucional puede fallar,
    # se intenta hasta que se consiga
    out_conv_features = None
    while(out_conv_features is None):
        conv_set, out_conv_features = getRandomConvSet(IMG_IN_CHANNELS)

    net_layers.extend(conv_set)

    net_layers.append({"layer_type": "Flatten"})

    linear_set, out_linear_features = getRandomLinearSet(out_conv_features)
    net_layers.extend(linear_set)


    # Creacion de la ultima capa y su transformacion
    net_layers.append({"layer_type": "Linear", "in_features": out_linear_features, "out_features": 1})
    net_layers.append({"layer_type": "Sigmoid"})


    # Obtenemos el hash code y lo ponemos en el nombre
    ret_pair = (hyperparams, net_layers)
    hash_code = getHashCode(ret_pair)
    hyperparams["model_name"] = "RandModel_" + hash_code


    return ret_pair












def getRandomConvSet(in_features):
    """
    Devuelve la parte de la red dedicada a las convoluciones en un par con
    los siguientes elementos:

        convSet_list: lista con las capas ordenadas,

        out_features: Numero de salidas de la ultima capa

    """
    convSet_list = []
    to_transform_img_dims = np.array(IMG_SIZE)

    # Encadenamos pares aleatorios de 1 a 8 veces
    for i in range(randint(1,9)):
        # Generamos el nuevo par y lo añadimos a la lista
        new_pair = getRandomConvPair(in_features)
        convSet_list.extend(new_pair)

        # Aplicamos la transformacion a las dimensiones de la imagen
        #print("dims: ", to_transform_img_dims)
        to_transform_img_dims = to_transform_img_dims - (new_pair[0]["kernel_size"]-1)
        #print("-: ", to_transform_img_dims)

        to_transform_img_dims = (to_transform_img_dims / new_pair[1]["kernel_size"]).astype(np.int32)
        #print("/: ", to_transform_img_dims)


        # Obtenemos las out_features del par, las cuales seran las nuevas in_features
        in_features = new_pair[0]["out_channels"]

    # Comprobamos que no son demasiadas reduccones de la imagen
    if (to_transform_img_dims[0] <= 0) or (to_transform_img_dims[1] <= 0):
        out_features = None
    else:
        out_features = int(to_transform_img_dims[0] * to_transform_img_dims[1] * in_features)

    # Devolvemos
    return convSet_list, out_features


def getRandomConvPair(in_features):
    """
    Crea dos capas, una de ellas convolucional y la otra de pooling de forma
    aleatoria.

    Parametros aleatorios y aleatoriedad en el tipo de capa de pooling:

            "MaxPool2d": nn.MaxPool2d,
            "AvgPool2d": nn.AvgPool2d,
            "LPPool2d": nn.LPPool2d

    Kernel de pooling aleatorio entre 2 y 8
    """
    # Creamos la capa convolucional
    conv_layer = getRandomConv2D(in_features)
    # Creamos la capa de pooling
    pool_layer = getRandomPool2D()
    # Devolvemos
    return [conv_layer, pool_layer]


def getRandomConv2D(in_features):
    """
    Crea una capa convolucional 2D con parametros aleatorios y la devueleve.
    """
    # Random
    out_features = randint(1, 100)
    kernel_rand = int(choice([3,5,7])) # Evitamos los numpy.int64

    # Creamos el dict y lo devolvemos
    return {"layer_type": "Conv2d", "in_channels": in_features, "out_channels": out_features, "kernel_size": kernel_rand}


def getRandomPool2D():
    """
    Crea una capa de Pooling 2D aleatoria y la devuelve.

    Tipos de capas de pooling:

            "MaxPool2d": nn.MaxPool2d,
            "AvgPool2d": nn.AvgPool2d,
            "LPPool2d": nn.LPPool2d
    """
    choice_num = randint(0,3)
    if choice_num == 0:
        # Creamos una capa MaxPool2d
        return {"layer_type": "MaxPool2d", "kernel_size": randint(2,9)}
    elif choice_num == 1:
        # Creamos una capa AvgPool2d
        return {"layer_type": "AvgPool2d", "kernel_size": randint(2,9)}
    elif choice_num == 2:
        # Creamos una capa LPPool2d
        return {"layer_type": "LPPool2d", "norm_type": randint(1, 100), "kernel_size": randint(2,9)}






def getRandomLinearSet(in_features):
    """
    Crea el set de capas densas que vienen despues de la convolucion.

        - in_features: Entradas que le vienen de la capa anterior
                       (numero de salidas de la ultima capa convolucional)

    Esta funcion no genera una ultima capa acorde a la salida que debiera tener
    la red, por lo que deberia de hacerse manualmente despues.

    Devuelve el par:  lista con el set de capas y out_features
    """
    linearSet_list = []

    for i in range(randint(1,11)):
        # Creamos el nuevo par
        new_pair = getRandomLinear(in_features)
        linearSet_list.extend(new_pair)

        # Obtenemos las out_features del par, las cuales seran las nuevas in_features
        in_features = new_pair[0]["out_features"]

    # Devolvemos
    return linearSet_list, in_features


def getRandomLinear(in_features):
    """
    Devuelve un par: capa Linear de parametros aleatorios y su transformacion no
    lineal acorde (preestablecida)

    Transformacion no lineal actual:  Sigmoid
    """
    # Creamos la capa linear
    linear_layer = {"layer_type": "Linear", "in_features": in_features, "out_features": randint(1, 200)}
    # Creamos la transformacion no lineal
    nol_trans = {"layer_type": "Sigmoid"}
    # Devolvemos
    return [linear_layer, nol_trans]







def getHashCode(data_pair):
    """
    Produce un hash code unico para cada combinacion de
    pares (hyperparams, net_layers).
    """
    hyperparams = data_pair[0]
    net_layers = data_pair[1]

    return sha256({"hyperparams": hyperparams, "net_layers": net_layers})


def printDict(d):
    """ Imprime un dict con tabulaciones """
    txt = json.dumps(d, indent=4)
    print(txt)


def printNetLayers(net_layers):
    """ Imprime el dict de las capas de la red visualmente mejor """
    for layer in net_layers:
        if layer["layer_type"] == "Flatten":
            print()

        print(layer)

        if layer["layer_type"] == "Flatten":
            print()
