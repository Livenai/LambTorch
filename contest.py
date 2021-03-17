"""
En este archivo se recogen las funciones que permitiran al usuario
inicializar el concurso de redes, crearlas usando los generadores aleatorios,
entrenarlas, ordenarlas y guardarlas.
"""
from colored import fg
import signal
import CL_trainer
from CL_trainer import CL_Trainer
import hyperparams_generator
from hyperparams_generator import getRandomHyperParamsV1, printDict, printNetLayers




def CtrlC_signal_handler(sig, frame):
    """
    Esta funcion se ejecuta cada vez que el usuario utiliza la
    combinacion Ctrl+C
    """
    print("\n\n\n\n\n\n\n\n")
    print('=========================  Closing Program  =========================')
    exit(0)
# Conectamos la señal con la funcion
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
        return net.obtenerValidationLoss()


def printRanking(net_list, verbose = False, just_return_str = False):
    """
    Musetra el ranking pasado por parametros. El orden de muestra es el orden
    de la estructura que reciba, por lo que debera estar ordenada previamente.

        [1º, 2º, 3º, 4º, ..., ultimo]

    Si verbose esta activo mostrara gran cantidad de informacion de cada una
    de las redes.

    Si just_return_str esta activo, en vez de imprimir el ranking, lo devuelve
    como string.
    """
    ret = ""

    ret += "\n╔════════════════════════════════════  "
    ret += fg(118) + "Ranking" + fg(15)
    ret += "  ════════════════════════════════════╗\n\n"

    for i, net in enumerate(net_list):# ╔ ╚ ╩ ╦ ╠ ═ ╬ ╝ ╗ ║ ╣
        if just_return_str:
            clor = ""
            net.color = clor
        else:
            clor = fg((i%77)+154)
            net.color = clor
        ret += "╔══  " + clor + str(i+1) + fg(15)
        ret += "º  ════════════════════════════════════════════════════════════════════════\n"
        ret += str(net) + "\n"


    if just_return_str:
        return ret
    else:
        print(ret)



def createRandomNetPool(pool_size):
    """
    Crea un conjunto de redes inicializadas aleatoriamente y las devueve en
    una lista.
    """
    ret = []
    # Para cada red
    for i in range(pool_size):
        # Creamos los hiperparametros aleatorios
        hyperparams, net_layers = getRandomHyperParamsV1()

        # Creamos la red con dichos parametros
        ret.append(CL_Trainer(hyperparams, net_layers))

    return ret



def trainNetPool(net_pool):
    """
    Realiza el entrenamiento de un conjunto de redes ya inicializadas.

    Modifica la lista, es decir, despues de usarse este metodo, la lista pasara
    a tener las redes ya entrenadas en su interior, en el mismo orden.
    """
    for net in net_pool:
        net.iniciarEntrenamiento()


def sortNetPool(net_pool, criterio=None):
    """
    Ordena el conjunto de redes de mejor a peor.

    Devuelve la lista ordenada. NO modifica la pasada por parametros.

    Si criterio es None, utiliza el criterio por defecto.
    """
    if criterio is None:
        net_pool.sort(key=evaluate_net)
        return net_pool
    else:
        net_pool.sort(key=criterio)
        return net_pool




def trainNetsForTestingRanking(num_nets):
    # Creamos una lista de redes
    train_list = createRandomNetPool(num_nets)

    # Entrenamos todas las redes
    trainNetPool(train_list)

    # Ordenamos el ranking
    train_list = sortNetPool(train_list)

    # Mostramos el ranking
    printRanking(train_list)
