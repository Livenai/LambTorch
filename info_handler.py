"""
En este script se guardaran las utilidades necesarias para el control de
informacion de las redes:

    - Cargar y guardar redes en formato json
    - Gestion de la lista de redes ya creadas
    - Transformar una red en un json con su info
    - Crear una red a partir de un json con la info







Despues de eso, modificar contest.py para que genere redes aleatorias, las entrene y las acumule en
el gran json, transformandolas a ese formato. Tienen que ser dos funciones diferentes, es decir, tiene que
haber una que sirva para añadir task al gran json (redes que aun no han sido entrenadas y por tanto no tienen metricas),
y debe haber otra funcion que recorra el json en busca de las task (redes no entrenadas) y las entrene, actualizando sus
datos y guardando las metricas del entrenamiento.

Hay que capturar el Ctrl-C para que si sucede, pare el entrenamiento que este haciendo y guarde los datos que tenia
resueltos hasta ese momento. aun que esto suponga perder los datos de la red que se estaba entrenando, al menos se guardan
los datos de las que ya se habian hecho.

Despues de esto deberiamos poder cargar el gran json y leerlo para entrenar las redes, asi como generar nuevas y acumularlas
para entrenar despues.

En contest deberia de haber dos funciones, una para entrenar las task, y otra para aladir nuevas task, pero es probable que
se necesite otra funcion para poder entrenar y generar nuevas redes automaticamente cuando haga falta.

por ultimo, habria que modificar el visor del ranking para que muestre todas las redes ordenadas con muuuuuuy pocos
datos, que muestre solo las ultimas N redes con algo mas de detalle o que muestre las ultimas N con todos los detalles.

Acordarse de arreglar lo de la memoria de la gpu, es decir, que se pueda limpiar la memoria despues de cada
entrenamiento, para poder entrenar de seguido, si no, se acaba la memoria y lanza una excepcion.
(o quizas es que hay que controlar si una red no cabe en la gpu)

"""
import json
import os
from datetime import datetime
from CL_trainer import CL_Trainer


parent_folder = os.path.abspath(os.path.dirname(__file__))



def loadTheBigTaskJson():
    """
    Carga el json que contiene las tareas pendientes, es decir, las redes
    que aun NO han sido entrenadas.

    Carga el archivo en la ruta /train_data/the_big_task_json.json
    """
    # Obtenemos la ruta
    train_data_path = os.path.join(parent_folder, "train_data")

    # Comprobamos si la ruta existe
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    # Comprobamos si el archivo existe
    json_path = os.path.join(train_data_path, "the_big_task_json.json")

    if os.path.exists(json_path):
        # Si existe, lo cargamos y lo devolvemos
        json_file = open(json_path)
        return json.load(json_file)

    else:
        # Si NO existe, creamos uno vacio y lo devolvemos
        return {}

def loadTheBigTrainedJson():
    """
    Carga el json que contiene las tareas completadas, es decir, las redes
    que YA han sido entrenadas y, por tanto, tienen metricas.

    Carga el archivo en la ruta /train_data/the_big_trained_json.json
    """
    # Obtenemos la ruta
    train_data_path = os.path.join(parent_folder, "train_data")

    # Comprobamos si la ruta existe
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    # Comprobamos si el archivo existe
    json_path = os.path.join(train_data_path, "the_big_trained_json.json")

    if os.path.exists(json_path):
        # Si existe, lo cargamos y lo devolvemos
        json_file = open(json_path)
        return json.load(json_file)

    else:
        # Si NO existe, creamos uno vacio y lo devolvemos
        return {}


def saveTheBigTaskJson(big_json):
    """
    Guarda el json que contiene las tareas pendientes, es decir, las redes
    que aun NO han sido entrenadas.

    Guarda el archivo en la ruta /train_data/the_big_task_json.json
    """
    # Obtenemos la ruta
    train_data_path = os.path.join(parent_folder, "train_data")

    # Comprobamos si la ruta existe
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    # Lo guardamos en la ruta del archivo
    json_path = os.path.join(train_data_path, "the_big_task_json.json")

    json_file = open(json_path, 'w')
    json.dump(big_json, json_file)


def saveTheBigTrainedJson(big_json):
    """
    Guarda el json que contiene las tareas completadas, es decir, las redes
    que YA han sido entrenadas y, por tanto, tienen metricas.

    Guarda el archivo en la ruta /train_data/the_big_trained_json.json
    """
    # Obtenemos la ruta
    train_data_path = os.path.join(parent_folder, "train_data")

    # Comprobamos si la ruta existe
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    # Lo guardamos en la ruta del archivo
    json_path = os.path.join(train_data_path, "the_big_trained_json.json")

    json_file = open(json_path, 'w')
    json.dump(big_json, json_file)

def CL_trainer2json(trainer):
    """
    Crea y devuelve un json que describe el objeto CL_trainer.
    """

    ret = {}
    # Comprobamos si el modelo ha sido entrenado
    if trainer.trained:
        # Si ha sido entrenado:
        # Guardamos lo inprescindible
        ret["net_layer_struct"] = trainer.net_layer_struct
        ret["hyperparams"] = trainer.hyperparams

        # Guardamos las metricas
        ret["history"] = trainer.history

        # Guardamos la info extra
        ret["trained"] = trainer.trained
        ret["num_params"] = trainer.num_params
        ret["nan_or_inf"] = trainer.nan_or_inf
        ret["creation_date"] = trainer.creation_date
        ret["model_name"] = trainer.model_name
        ret["net_train_loss"] = trainer.net_train_loss
        ret["net_train_accuracy"] = trainer.net_train_accuracy
        ret["net_val_loss"] = trainer.net_val_loss
        ret["net_val_accuracy"] = trainer.net_val_accuracy


        now = str(datetime.now())
        ret["last_modification_date"] = now[:now.rfind(".")]

    else:
        # Si NO ha sido entrenada:
        # Guardamos lo imprescindible
        ret["net_layer_struct"] = trainer.net_layer_struct
        ret["hyperparams"] = trainer.hyperparams

        # Guardamos info extra
        ret["trained"] = trainer.trained
        ret["creation_date"] = trainer.creation_date
        ret["model_name"] = trainer.model_name

        now = str(datetime.now())
        ret["last_modification_date"] = now[:now.rfind(".")]


    return ret



def json2CL_trainer(json_data):
    """
    Crea y devuelve un objeto CL_trainer a partir de su version en json.
    """

    # Creamos un objeto nuevo
    ret_obj = CL_Trainer(json_data["hyperparams"], json_data["net_layer_struct"])

    # Cargamos los valores
    ret_obj.loadFromJson(json_data)

    # Devolvemos
    return ret_obj
