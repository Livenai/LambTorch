import signal

import CL_trainer
from CL_trainer import CL_Trainer
import hyperparams_generator
from hyperparams_generator import getRandomHyperParamsV1, printDict, printNetLayers



# PARAMETROS

NUM_NETWORKS = 2




# Captura de Ctrl+C
def CtrlC_signal_handler(sig, frame):
    print("\n\n\n\n\n\n\n\n")
    print('=========================  Closing Program  =========================')
    exit(0)

signal.signal(signal.SIGINT, CtrlC_signal_handler)



# Criterio de ordenacion del ranking
def evaluate_net(net):
    """
    Devuelve como puntuacion el val_accuracy, a no ser que la red
    tenga NaN o Inf, en cuyo caso devuelve una puntuacion muy penalizada.
    """
    if net.nan_or_inf:
        return 99999999
    else:
        return net.obtenerValidationAccuracy()







####################################  Start  ###################################

# Creamos una lista de redes
train_list = []

for i in range(NUM_NETWORKS):
    # Obtenemos los hiperparametros y la morfologia de la red
    hyperparams, net_layers = getRandomHyperParamsV1()

    """
    printDict(hyperparams)
    print("-----------------------------------------------------------")
    printNetLayers(net_layers)
    """
    # Creamos la instancia de entrenamiento
    train_list.append(CL_Trainer(hyperparams, net_layers))



# Entrenamos todas las redes
for train in train_list:
    train.iniciarEntrenamiento()


# Ordenamos el ranking
train_list.sort(key=evaluate_net)

# Mostramos el ranking
for t in train_list:
    print(t)
