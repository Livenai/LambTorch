from dynamic_dataset_loader import CLDL_r1
from CL_trainer import CL_Trainer
import torch
from torch_model_asker import ClasiLamb_2_2_ModelAsker, ClasiLamb_2_1_ModelAsker
import cv2
from dynamic_dataset_loader import CLDL_r1
from util import printProgressBar
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
import json
from colored import fg, attr
import numpy as np
from datetime import datetime


device = ("cuda" if torch.cuda.is_available() else "cpu")


dataset_path = "clasiLamb_2-2_CUS"
parent_folder = os.path.abspath(os.path.dirname(__file__))
raw_dataset_path = parent_folder
saving_path = os.path.join(parent_folder, "savings")

weight_model_path = "CL22_RandModel-ReLU_N3d80fcfd7c6567591cd7083b0211a8c70de8a866c9aeb891eab578467696c370"
lamb_model_path   = "CL21_RandModel_5debcadeffa6344a49f2b67aaa2e0884d23f501ec277d97bb8ee1ddf11d5e247"

OVERRIDE_MAXVALUE = 2000  # -1 = auto detect
max_out_img_num = -1


B = fg(15)
V = fg(33) # 45 azul


relative_epoch = datetime.utcfromtimestamp(0)





################################################################################
################################################################################
################################################################################
"""                              FUNCIONES                                   """
################################################################################
################################################################################
################################################################################


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    rects = bars
    """
    global BAR_NUMBER_SIZE
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,
                height,
                str(round(height, 1)),
                ha ='center',
                va ='bottom',
                size = 8)










################################################################################
################################################################################
################################################################################
"""                              ERROR PLOT                                  """
################################################################################
################################################################################
################################################################################



def getErrorPlot():

    # dataset (normalizamos las etiquetas entre 0 y 35 para los pesos
    dataset_loader = CLDL_r1(dataset_path, img_transform=None, manual_label_normalize= None)



    # Cargamos el modelo CNN
    asker = ClasiLamb_2_2_ModelAsker()
    asker.loadModel(weight_model_path)


    # Preparamos los Datos para hacer la plot. Para ello, leemos uno a uno todos los
    # elementos del data loader, guardamos el label y pasamos la imagen por la red.
    # Nos quedamos con el peso estimado tambien y lo guardamos todo en dos listas
    # separadas, pero cada indice relaciona el peso y la prediccion
    labels = []
    preds  = []
    j = 0
    for x, y in dataset_loader:
        prediction = asker.ask(x) * 35
        # print("label: ", y, "    pred: ", prediction, "         diff: ", abs(y-prediction))
        labels.append(y)
        preds.append(prediction)




        printProgressBar(j, len(dataset_loader), prefix='Obteniendo predicciones de la red: ', suffix='Completado',length=50,color=33)
        j += 1


        if j == 15211:
            break


    print(len(labels), " entradas en labels_and_preds.")
    print("y type: ", type(y))
    print("prediction type ", type(prediction))





    # Procesamos la estructura anterior creando las dos listas finales para la plot.
    # Estas son, una lista de labels (los kilogramos discretos, Ej: 14, 15, 16) y
    # una lista de valores, que seran la media de errores de los pesos que entren
    # dentro de esa categoria. Cada peso entra dentro de la categoria que tenga
    # mas cerca (Ej: en la categoria 16Kg entran desde 15.50 hasta 16.49)

    # Lista de valores discretos
    x_values = [i for i in range(int(round(min(labels), 0)), int(round(max(labels), 0))+1)]

    print(min(labels))
    print(max(labels))
    print(round(min(labels)))
    print(round(max(labels)))

    print(x_values)


    # Acumulador de errores en cada categoria de peso
    acum = [[] for i in range(len(x_values))]

    # Para cada peso en label, obtenemos su redondeo sin decimales, obtenemos el
    # idx de la categoria en la que iria y añadimos EL ERROR (diferencia) a la
    # lista de esa categoria
    for i, p in enumerate(labels):
        int_p = round(p, 0)
        cat_idx = x_values.index(int_p)
        acum[cat_idx].append(abs(p-preds[i]))

    print("Categoria  :  Nº de pesos en la categoria")
    for i in range(len(x_values)):
        print(x_values[i], " : ", len(acum[i]))




    # Metemos en el idx de cada categoria la media de pesos
    y_values = [[] for i in range(len(x_values))]

    for i in range(len(x_values)):
        try:
            y_values[i] = mean(acum[i])
        except:
            y_values[i] = 0

    print(y_values)



    # Montamos la plot
    color_list = ["deepskyblue", "aquamarine", "limegreen", "yellow", "gold", "darkorange", "lightcoral", "orchid", "mediumpurple"]

    num_lambs = 1

    fig, axs = plt.subplots(num_lambs)

    i = 0

    bars = axs.bar(x_values, y_values, color=color_list[5])
    autolabel(bars, axs)

    plt.sca(axs)
    plt.xticks(rotation=-40, ha="left", size=8)
    plt.xticks(x_values)
    plt.ylabel("Error [Kg]")
    plt.title("Error segun el peso", size = 20)

    plt.show()













################################################################################
################################################################################
################################################################################
"""                             DAILY AVG PLOT                               """
################################################################################
################################################################################
################################################################################




def getDailyAvgPlot():

    num_images = 0

    # contamos las imagenes totales
    for json_number, json_name in enumerate(glob.glob(os.path.join(saving_path, '*.json'))):

        f = open(json_name)
        j = json.load(f)

        i = 0
        for key in j:
            num_images = num_images + 1
            i += 1
        print("Numero del Json: " + V + str(json_number) + B)
        print("Imagenes en el Json: " + V + str(i) + B, end='\n\n')



    print(attr(4) + "Imagenes Totales:" + attr(0) + " " + V + str(num_images) + B, end='\n\n')



    i2 = 0
    label_list = []
    num_lamb = 0
    num_no_lamb = 0
    num_fail = 0
    daily_weight = {}

    asker = ClasiLamb_2_1_ModelAsker()
    asker.loadModel(os.path.join(parent_folder, lamb_model_path))

    # para todos los .json de labels
    for json_number, json_name in enumerate(glob.glob(os.path.join(saving_path, '*.json'))):

        f = open(json_name)
        j = json.load(f)
        json_key_name = json_name[json_name.rfind("/")+1:json_name.rfind(".")-1]
        daily_weight[json_key_name] = []


        # estructuramos las imagenes del json
        for key in j:
            # Comprobacion de maximo de imagenes. -1 para que no haya limite
            if max_out_img_num >= 0:
                if num_lamb == max_out_img_num:
                    break

            # cargamos cada imagen
            img_path = raw_dataset_path + str(j[key]["path_depth_default_image"])
            img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH)

            # comprobamos si la imagen existe y ha sido cargada correctamente (si no, saltamos a la siguiente)
            try:
                if len(img) == 0:
                    print("[!] Error al cargar la imagen: " + V + str(img_path) + B)
                    exit()
            except:
                num_fail += 1
                i2 += 1
                continue

            # preparamos la imagen para ser normalizada posteriormente (todo pixel que supere OVERRIDE_MAXVALUE se queda en OVERRIDE_MAXVALUE)
            if OVERRIDE_MAXVALUE != -1:
                img = np.clip(img, None, OVERRIDE_MAXVALUE)

            """
            # le hacemos crop
            x = 38
            y = 102
            h = 230
            w = 510
            img = img[y:y + h, x:x + w]
            """


            # invertimos la imagen
            # if OVERRIDE_MAXVALUE != -1:
            #   img = OVERRIDE_MAXVALUE-img


            # Normalizamos
            #img = img/OVERRIDE_MAXVALUE

            # Preguntamos a la red si la imagen es correcta
            # Si es correcta, la red sacara un 0, si no, sacara un 1
            res, val = asker.ask(img)
            if res == 1.0:
                is_correct_lamb = False
            elif res == 0.0:
                is_correct_lamb = True

            # Si la imagen es correcta guardamos la imagen y la etiqueta, si no, descartamos
            if is_correct_lamb:


                # Acumulamos el peso en su dia, en funcion del nombre del json
                daily_weight[json_key_name].append(float(j[key]["weight"]))



                num_lamb += 1

            else:
                num_no_lamb += 1


            i2 += 1
            printProgressBar(i2, num_images, prefix='Estructurando JSON ' + str(json_number) + ':', suffix='Completado',length=50,color=33)



    # Procesamos el json en dos listas para pasarselo a la plot

    x_values = [str(key) for key in daily_weight.keys()]
    x_values.sort(key = lambda v: datetime.strptime(v, '%Y-%m-%d'), reverse=False)


    y_values = []
    for k in x_values:
        try:
            y_values.append(mean(daily_weight[k]))
        except:
            y_values.append(0)




    # Montamos la plot
    color_list = ["deepskyblue", "aquamarine", "limegreen", "yellow", "gold", "darkorange", "lightcoral", "orchid", "mediumpurple"]

    num_lambs = 1

    fig, axs = plt.subplots(num_lambs)

    i = 0

    bars = axs.bar(x_values, y_values, color=color_list[2])
    autolabel(bars, axs)

    plt.sca(axs)
    plt.xticks(rotation=-40, ha="left", size=8)
    plt.xticks(x_values)
    plt.ylabel("Peso [Kg]")
    plt.title("Peso Medio Diario", size = 20)

    plt.show()






    y_values = []
    for k in x_values:
        y_values.append(len(daily_weight[k]))

    # Montamos la plot de cantidades
    color_list = ["deepskyblue", "aquamarine", "limegreen", "yellow", "gold", "darkorange", "lightcoral", "orchid", "mediumpurple"]

    num_lambs = 1

    fig, axs = plt.subplots(num_lambs)

    i = 0

    bars = axs.bar(x_values, y_values, color=color_list[1])
    autolabel(bars, axs)

    plt.sca(axs)
    plt.xticks(rotation=-40, ha="left", size=8)
    plt.xticks(x_values)
    plt.ylabel("Numero de Imagenes")
    plt.title("Numero de Imagenes Diarias", size = 20)

    plt.show()




























################################################################################
################################################################################
################################################################################
"""                                  MAIN                                    """
################################################################################
################################################################################
################################################################################

getDailyAvgPlot()
