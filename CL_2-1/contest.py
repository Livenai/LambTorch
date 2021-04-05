"""
En este archivo se recogen las funciones que permitiran al usuario
inicializar el concurso de redes, crearlas usando los generadores aleatorios,
entrenarlas, ordenarlas y guardarlas.
"""
import time
from datetime import timedelta
from colored import fg
import signal
import CL_trainer
from CL_trainer import CL_Trainer
import hyperparams_generator
from hyperparams_generator import getRandomHyperParamsV1, printDict, printNetLayers
import info_handler
import telegram_debugger
from telegram_debugger import sendMSG
from emoji import emojize
import traceback



# PARAMETROS
GPU_BREAK_TIME = 5 * 60 # 5 mins de descanso entre entrenamientos


def CtrlC_signal_handler(sig, frame):
    """
    Esta funcion se ejecuta cada vez que el usuario utiliza la
    combinacion Ctrl+C
    """
    print("\n\n\n")
    sendMSG("Ctrl+C o SIGINT detectado, cerrando programa... ", is_error=True)
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


def __printRanking(net_list, just_return_str=False, num_nets_to_show=-1, colored_text=False, str_type="classic"):
    """
    Privado.

    Musetra el ranking pasado por parametros. El orden de muestra es el orden
    de la estructura que reciba, por lo que debera estar ordenada previamente.

        [1º, 2º, 3º, 4º, ..., ultimo]

    Si verbose esta activo mostrara gran cantidad de informacion de cada una
    de las redes.

    Si just_return_str esta activo, no imprime el ranking por pantalla.
    Solo lo devuelve como str.

    Si num_nets_to_show vale -1, se mostraran TODAS las redes. En otro caso
    se mostraran solo las  num_nets_to_show  mejores redes.

    El parametro  str_type  define el tipo de impresion.
    Puede ser:
            "classic" - Clasico. Info en lineas normales.
            "tab" - Tabulado. Cada dato en una tabulacion mas.
            "extended" - Extendido. Muestra mucha info de cada red.
    """
    # Cabecera
    ret = ""
    no_color_ret = ""

    ret += "\n╔════════════════════════════════════  "
    ret += fg(118) + "Ranking" + fg(15)
    ret += "  ════════════════════════════════════╗\n\n"

    no_color_ret += "\n╔════════════════════════════════════  "
    no_color_ret += "Ranking"
    no_color_ret += "  ════════════════════════════════════╗\n\n"


    # Cuerpo del ranking, mas extenso o no en funcion de verbose
    if str_type == "classic":
        # Informacion ligera de las redes
        for i, net in enumerate(net_list):# ╔ ╚ ╩ ╦ ╠ ═ ╬ ╝ ╗ ║ ╣
            if i == num_nets_to_show:
                break
            clor = fg((i%77)+154)
            net.printColor = clor
            net.resetColor = fg(15)

            ret += "╔══  " + clor + str(i+1) + fg(15)
            ret += "º  ════════════════════════════════════════════════════════════════════════\n"
            ret += str(net) + "\n"

            net.printColor = ""
            net.resetColor = ""

            no_color_ret += "╔══  " + str(i+1)
            no_color_ret += "º  ════════════════════════════════════════════════════════════════════════\n"
            no_color_ret += str(net) + "\n"

    elif str_type == "tab":
        # Informacion ligera de las redes
        for i, net in enumerate(net_list):# ╔ ╚ ╩ ╦ ╠ ═ ╬ ╝ ╗ ║ ╣
            if i == num_nets_to_show:
                break
            clor = fg((i%77)+154)
            net.printColor = clor
            net.resetColor = fg(15)

            ret += "╔══  " + clor + str(i+1) + fg(15)
            ret += "º  ════════════════════════════════════════════════════════════════════════\n"
            ret += str(net.getTabuledSTR()) + "\n"

            net.printColor = ""
            net.resetColor = ""

            no_color_ret += "╔══  " + str(i+1)
            no_color_ret += "º  ════════════════════════════════════════════════════════════════════════\n"
            no_color_ret += str(net.getTabuledSTR()) + "\n"

    elif str_type == "extended":
        # Info extendida de cada red
        for i, net in enumerate(net_list):# ╔ ╚ ╩ ╦ ╠ ═ ╬ ╝ ╗ ║ ╣
            if i == num_nets_to_show:
                break
            clor = fg((i%77)+154)
            net.printColor = clor
            net.resetColor = fg(15)

            ret += "╔══  " + clor + str(i+1) + fg(15)
            ret += "º  ════════════════════════════════════════════════════════════════════════\n"
            ret += str(net.getExtendedSTR()) + "\n"

            net.printColor = ""
            net.resetColor = ""

            no_color_ret += "╔══  " + str(i+1)
            no_color_ret += "º  ════════════════════════════════════════════════════════════════════════\n"
            no_color_ret += str(net.getExtendedSTR()) + "\n"

    else:
        raise Exception("Parametro str_type invalido en la funcion __printRanking:  " + str(str_type))



    if just_return_str:
        if colored_text:
            return ret
        else:
            return no_color_ret
    else:
        if colored_text:
            print(ret)
        else:
            print(no_color_ret)




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
    """
    Entrenamos un numero concreto de redes solo para testear que las diversas
    operaciones funcionan correctamente.
    """
    # Creamos una lista de redes
    train_list = createRandomNetPool(num_nets)

    # Entrenamos todas las redes
    trainNetPool(train_list)

    # Ordenamos el ranking
    train_list = sortNetPool(train_list)

    # Transformamos la lista a texto en json y viceversa
    new_list = []
    for net in train_list:
        json_data = info_handler.CL_trainer2json(net)
        new_obj = info_handler.json2CL_trainer(json_data)

        new_list.append(new_obj)

    # Mostramos el ranking
    printRanking(train_list)

    # Mostramos el ranking obtenido de hacer la transformacion a json y
    # viceversa. Deberia ser igual al ranking anterior
    printRanking(new_list)


def readOneTask():
    """
    Carga el gran json de tareas, extrae una tarea y lo guarda.

    Devuelve la tarea transformada en CL_trainer.
    Si no quedan tareas, devuelve None
    """
    # Cargamos el gran json de tareas
    big_task_json = info_handler.loadTheBigTaskJson()

    # Informamos de la cantidad de tareas que quedan
    sendMSG("Quedan  " + str(len(big_task_json)) + "  tareas por hacer.")

    # Comprobamos si quedan tareas para devolver
    if len(big_task_json) > 0:
        # Seleccionamos una tarea
        ret_key = list(big_task_json.keys())[0]

        # Extraemos la tarea (eliminandola del json)
        ret_jsoned_net = big_task_json.pop(ret_key)

        # Guardamos el json
        info_handler.saveTheBigTaskJson(big_task_json)

        # Devolvemos la tarea
        return info_handler.json2CL_trainer(ret_jsoned_net)
    else:
        sendMSG("No quedan tareas por hacer en el gran json de tareas", is_warning=True)
        return None


def saveOneDoneTask(task):
    """
    Guarda una tarea completada en el gran json de entrenamientos realizados.

    Si el json no existe, lo crea y mete la tarea completada dentro.
    """
    # Cargamos el gran json de entrenamientos realizados
    big_trained_json = info_handler.loadTheBigTrainedJson()

    # Convertimos la tarea a formato json
    jsoned_task = info_handler.CL_trainer2json(task)

    # Acumulamos la tarea en el gran json de entrenamientos realizados
    net_name = jsoned_task["model_name"]
    big_trained_json[net_name] = jsoned_task

    # Guardamos el json
    info_handler.saveTheBigTrainedJson(big_trained_json)



def generateTasks(num_tasks):
    """
    Genera un numero dado de redes preparadas para iniciar su entrenamiento y
    las guarda como tareas en el gran json de tareas.
    """
    # Generamos un pool de redes
    nets_list = createRandomNetPool(num_tasks)

    # Las transformamos a formato json
    json_list = [info_handler.CL_trainer2json(x) for x in nets_list]

    # Cargamos el gran json de tareas
    big_task_json = info_handler.loadTheBigTaskJson()

    # Acumulamos las tareas
    for jsoned_task in json_list:
        net_name = jsoned_task["model_name"]
        big_task_json[net_name] = jsoned_task # ¿hacemos copia antes?

    # Guardamos el gran json con las nuevas tareas acumuladas
    info_handler.saveTheBigTaskJson(big_task_json)


def trainTask():
    """
    Obtiene una tarea del gran json de tareas y la entrena. Posteriormente
    guarda la red entrenada en el gran json de entrenamientos realizados.

    Si no quedan tareas por hacer, lanza un mensaje
    de advertencia y devuelve None.

    Devuelve 0 en otro caso.

    Si no se puede entrenar la red por que no cabe en la gpu, envia un mensaje
    y termina. Descarta la red.
    """
    # Obtenemos una tarea
    task = readOneTask()

    if task is None:
        # Notificamos que no quedan tareas por hacer y paramos
        sendMSG("No quedan tareas por hacer", is_warning=True)
        return None

    # Entrenamos la tarea
    try:
        trainNetPool([task])
    except:
        sendMSG("La red no cabe en la GPU. Descartando...", is_warning=True)
        sendMSG(traceback.format_exc(), is_error=True)

        return 0


    # Si la red no tienen NaN o Inf, guardamos los resultados
    if task.nan_or_inf == False:
        saveOneDoneTask(task)

    return 0


def trainRemainingTasks():
    """
    Inicia el entrenamiento de todas las tareas que queden por realizar.
    """
    # mientras queden tareas, entrenamos
    control = 0
    i = 0
    # Cantidad de cuadrados
    s = 6
    while control is not None:
        sendMSG(getHashedSquares(i,s) + "  RED  " + str(i) + "  " + getHashedSquares(i,s))
        control = trainTask()
        sendMSG("Dando un descanso a la GPU de " + str(timedelta(seconds=GPU_BREAK_TIME)) + "\n\n")
        time.sleep(GPU_BREAK_TIME)
        i += 1

    # No quedan tareas por realizar
    sendMSG("Se han acabado todas las tareas con exito.")



def showRanking(num_nets_to_show = -1):
    """
    Muestra el ranking usando el gran json de
    entrenamientos realizados.

    En el futuro se usara un diccionario ordenado, no teniendo que ordenar
    al cargar el json.
    """
    # Cargamos el gran json
    big_trained_json = info_handler.loadTheBigTrainedJson()

    # Transformamos a lista de CL_trainer
    net_list = []
    for key in big_trained_json:
        net = info_handler.json2CL_trainer(big_trained_json[key])
        net_list.append(net)

    # Ordenamos la lista
    sorted_net_list = sortNetPool(net_list)

    # Mostramos el ranking por pantalla con colores
    __printRanking(sorted_net_list, colored_text=True, num_nets_to_show=num_nets_to_show)

    # Obtenemos el ranking como str sin colores
    str_ranking = __printRanking(sorted_net_list, just_return_str=True, num_nets_to_show=num_nets_to_show, str_type="tab")

    # Enviamos ranking al admin
    sendMSG("Mostrando ranking de " + str(len(sorted_net_list)) + " redes:")
    sendMSG(str_ranking, dont_print=True)


def sendFinalMSG():
    """
    realiza el envio y muestra el mensaje final. Indicando que todo
    ha finalizado correctamente.
    """
    msg = ""

    for i in range(6):
        msg += ":white_large_square:"

    msg += "  Ejecucion terminada  "

    for i in range(6):
        msg += ":white_large_square:"

    sendMSG(msg)


def sendStarMSG():
    """
    Realiza el envio de un mensaje inicial.
    """
    msg = ""

    for i in range(6):
        msg += ":black_square_button:"

    msg += "  Inicio de entrenamiento  "

    for i in range(6):
        msg += ":black_square_button:"

    sendMSG(msg)


def getHashedSquares(hash_id, amount):
    """
    Devuelve una tira de cuadrados (emojis) de un color, en funcion de  hash_id.
    Devuelve la cantidad de cuadrados dado por el parametro  amount.
    """
    # Creamos la lista de candidatos
    square_list = [
        emojize(":red_square:"),
        emojize(":orange_square:"),
        emojize(":yellow_square:"),
        emojize(":green_square:"),
        emojize(":blue_square:"),
        emojize(":purple_square:"),
        emojize(":brown_square:"),
        emojize(":black_large_square:")
    ]

    # Obtenemos un candidato en funcion de  hash_id
    selected_square = square_list[hash_id % len(square_list)]

    # Repetimos el candidato la cantidad dada por  amount
    ret = ""
    for i in range(amount):
        ret += selected_square

    # Lo devolvemos
    return ret
