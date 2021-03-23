import torch
from torch import nn
import traceback
import copy

# Test de la clase al ser importado o ejecutado este archivo .py
DEBUG_TEST = False


class CL_CustomModel(nn.Module):
    """ ClasiLamb Custom Model v1"""

    def __init__(self, original_net_layer_list, device, params_datatype="float"):
        """
        Ctor. de la clase. Recibe como parametros un dictionary (json), lista o
        tupla con la morfologia y parametrizacion de la red a construir.

        Tambien recibe como parametros:

            - device: String con el tipo de dispositivo donde se ejecutara la red.
                      Puede ser "cuda" o "cpu", entre otros.

            - params_datatype: String con el tipo de datos en el que se almacenaran
                               los parametros entrenables de la red.
                               Puede ser "float", "double" o "int".

        La estructura de la variable de entrada net_layer_list es la siguiente:

            - Si es un dict, claves y valores deben estar ordenados con la
              morfologia de la red:

            net_layer_list{
                0: layer_dictionary,
                1: layer_dictionary,
                2: layer_dictionary,
                3: layer_dictionary,
                ...
            }

            - Si es una lista o tupla ordenada con la morfologia de la red:

            [
                layer_dictionary,
                layer_dictionary,
                layer_dictionary,
                layer_dictionary,
            ...
            ]


        Cada elemento layer_dictionary debe contener las palabras clave del tipo
        de capa que es y los parametros que debe llevar.
        Todos deben llevar la palabra clave "layer_type" con un String indicando el
        tipo de capa que es.
        Se puede consultar el tipo de capas aceptadas en la funcion getLayerTypeDict().
        Acepta funciones sin parametros.

            layer_dictionary{
                layer_type: string,
                layer_param1: int,
                layer_param2: float,
                layer_param3: bool,
                ...
            }



        Recordar que el parametro in_channels de la primera capa debe ser
        igual al numero de canales de la imagen de entrada:

        in_channels=img_channels

        """

        # Primero el super de la clase heredada
        super(CL_CustomModel, self).__init__()

        # Cargamos el diccionario con los tipos de capas aceptadas en self.layerTypeDict
        self.readLayerTypeDict()

        # Variables de clase
        self.called_func_list = []

        # Clonamos el dict para poder modificarlo
        net_layer_list = copy.deepcopy(original_net_layer_list)

        # Comprobamos si la variable de entrada es una lista, tupla o un dict
        if net_layer_list is list or tuple:
            self.container = net_layer_list
        elif net_layer_list is dict:
            self.container = net_layer_list.values()


        # Recorremos el contenedor
        for layer_dictionary in self.container:
            # Preparamos los parametros
            in_params = layer_dictionary
            layer_type = in_params.pop("layer_type", None)
            func_to_call = self.layerTypeDict[layer_type]

            # Llamamos a la funcion del tipo de capa correspondiente
            try:
                self.called_func_list.append(func_to_call(**in_params))
            except Exception as e:
                print("[!] Parece que algo ha salido mal al crear una de las capas de la red (", str(layer_type),"):\n")
                print(traceback.format_exc())



        # Creamos el contenedor Sequential con todas las capas
        self.sequential_container = nn.Sequential(*self.called_func_list)


        # Lo transformamos y movemos acorde a los parametros de entrada
        self.sequential_container = self.sequential_container.to(device)
        if params_datatype == "float":
            self.sequential_container = self.sequential_container.float()
        elif params_datatype == "int":
            self.sequential_container = self.sequential_container.int()
        elif params_datatype == "double":
            self.sequential_container = self.sequential_container.double()





    def forward(self, x):
        x = self.sequential_container(x)
        return x





    def readLayerTypeDict(self):
        a = {
        # Capas convolucionales
        "Conv2d": nn.Conv2d,

        # Capas de pooling
        "MaxPool2d": nn.MaxPool2d,
        "AvgPool2d": nn.AvgPool2d,
        "LPPool2d": nn.LPPool2d,
        "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
        "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,

        # Capas de padding
        "ReflectionPad2d": nn.ReflectionPad2d,
        "ReplicationPad2d": nn.ReplicationPad2d,
        "ZeroPad2d": nn.ZeroPad2d,
        "ConstantPad2d": nn.ConstantPad2d,

        # Capas de activaciones no lineales
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "Softmax": nn.Softmax,
        "Linear": nn.Linear,

        # Capas de normalizacion
        "BatchNorm2d": nn.BatchNorm2d,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "LayerNorm": nn.LayerNorm,

        # Capas de dropout
        "Dropout": nn.Dropout,
        "Dropout2d": nn.Dropout2d,
        "AlphaDropout": nn.AlphaDropout,

        # Capas de utilidad
        "Flatten": nn.Flatten
        }

        # Variable de clase
        self.layerTypeDict = a















if DEBUG_TEST == True:
    print("-----  inicio del test -----")

    device = "cpu"

    net_layers = [
            {"layer_type": "Conv2d", "in_channels": 1, "out_channels": 20, "kernel_size": 3},
            {"layer_type": "MaxPool2d", "kernel_size": (2,2)},
            {"layer_type": "Conv2d", "in_channels": 20, "out_channels": 50, "kernel_size": 3},
            {"layer_type": "MaxPool2d", "kernel_size": (2,2)},
            {"layer_type": "Conv2d", "in_channels": 50, "out_channels": 100, "kernel_size": 3},
            {"layer_type": "MaxPool2d", "kernel_size": (2,2)},
            {"layer_type": "Conv2d", "in_channels": 100, "out_channels": 200, "kernel_size": 3},
            {"layer_type": "MaxPool2d", "kernel_size": (2,2)},

            {"layer_type": "Flatten"},

            {"layer_type": "Linear", "in_features": 200*28*38, "out_features": 500},
            {"layer_type": "Sigmoid"},
            {"layer_type": "Linear", "in_features": 500, "out_features": 200},
            {"layer_type": "Sigmoid"},
            {"layer_type": "Linear", "in_features": 200, "out_features": 50},
            {"layer_type": "Sigmoid"},
            {"layer_type": "Linear", "in_features": 50, "out_features": 1},
            {"layer_type": "Sigmoid"}

    ]
    print("creacion de net_layer_list")

    CM = CL_CustomModel(net_layers, device)
    print("creacion de la instancia de la clase customModel")

    print("\n")
    print(CM)



    print("-----  fin del test -----")
