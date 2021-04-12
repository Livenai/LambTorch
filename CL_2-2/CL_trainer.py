"""
Esta clase engloba el resto de clases y codigo necesario para entrenar redes,
por lo que puede realizarse UN entrenamiento completo de UNA red con cada
instancia de esta clase.
"""



"""
ejemplo de net_layer_list:

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


Ejemplo de hiperparametros

        learning_rate = 0.00001
        batch_size = 1
        epochs = 50

        training_percent = 0.9 # 90% de imagenes para entrenamiento
        shuffle = True
        pin_memory = True
        num_workers = 1
        model_name = "CL21_M1_50ep"

"""

import os, enlighten
import numpy as np
from colored import fg
from math import isnan, isinf
from datetime import datetime
from copy import deepcopy

import torch
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
device = ("cuda" if torch.cuda.is_available() else "cpu")

import dynamic_dataset_loader
from plot import show_plot_and_save
from custom_model import CL_CustomModel
import telegram_debugger
from telegram_debugger import sendMSG



class CL_Trainer():
    """
    Clase que representa un entrenamiento.

    Esta clase permite preparar los hiperparametros y propiedades de la red,
    entrenarla y obtener los resultados



    Para mas informacion sobre la forma que net_layer_struct
    debe tener, mirar el comentario del constructor de CL_CustomModel
    en custom_model.
    """

    def __init__(self, hyperparams, net_layer_struct):
        """
        Ctor.

        Necesita como parametros:

            - hyperparams: Dict con los hiperparametros siguiendo la siguiente
                           nomenclatura:

                           learning_rate: float
                           batch_size: int
                           epochs: int
                           training_percent: float de 0.0 a 1.0
                           model_name: String
        """
        # Guardamos los parametros como variables de clase
        # Forma de la red
        self.net_layer_struct = net_layer_struct

        # Parametros
        self.hyperparams = hyperparams
        self.learning_rate = hyperparams["learning_rate"]
        self.batch_size = hyperparams["batch_size"]
        self.epochs = hyperparams["epochs"]
        self.training_percent = hyperparams["training_percent"]
        self.model_name = hyperparams["model_name"]

        # Parametros que no se tocan
        self.shuffle = True
        self.pin_memory = True
        self.num_workers = 1

        # Colores
        self.B = fg(15)
        self.C = fg(154)


        # SEED & Torch
        torch.manual_seed(42)
        torch.set_printoptions(edgeitems=3)

        # PATH
        self.parent_folder = os.path.abspath(os.path.dirname(__file__))
        self.train_data_path = os.path.join(self.parent_folder, "train_data")
        self.dataset_path = os.path.join(self.parent_folder, "clasiLamb_2-2_CUS")
        self.saved_models_path = os.path.join(self.train_data_path, "saved_models")
        self.plots_path = os.path.join(self.train_data_path, "plots")

        # Parametros extra
        self.model = None
        self.net_train_loss = None
        self.net_train_accuracy = None
        self.net_val_loss = None
        self.net_val_accuracy = None
        self.std_dev = None
        self.loss_hist = np.array([]) # Uso temporal
        self.history = {}
        self.trained = False
        self.num_params = None
        self.nan_or_inf = False
        self.printColor = ""
        self.resetColor = ""
        now = str(datetime.now())
        self.creation_date = now[:now.rfind(".")]
        self.last_modification_date = self.creation_date
        self.manual_label_normalize = 35.0 # valor maximo de las labels, para normalizar. None si no normalizamos

        # Aux para el early stopping
        self.best_test_loss = 999999999
        self.current_best_model = None


    def __str__(self):
        """
        Devuelve un String con el nombre de la red y las metricas si las tuviera.

        ESTILO CLASICO

        Solo colorea el String si printColor y resetColor son un color.
        """
        NUM_DECIMALS = 4

        name_color = ""

        if (self.resetColor != "") and (self.printColor != ""):
            name_color = fg(242)


        ret = name_color + self.model_name + self.resetColor
        ret += "  |\n\t|  "

        if self.trained:
            # Parametros
            ret += "num_params: " + self.printColor + str(self.num_params) + self.resetColor
            # Metricas
            ret += "  t_l: " + self.printColor + str(round(self.obtenerTrainLoss(), NUM_DECIMALS)) + self.resetColor
            ret += "  t_err_mean: " + self.printColor + str(round(self.obtenerTrainAccuracy(), NUM_DECIMALS)) + self.resetColor + " Kg"
            ret += "  val_l: " + self.printColor + str(round(self.obtenerValidationLoss(), NUM_DECIMALS)) + self.resetColor
            ret += "  val_err_mean: " + self.printColor + str(round(self.obtenerValidationAccuracy(), NUM_DECIMALS)) + self.resetColor + " Kg"
            ret += "  val_std_dev: ±" + self.printColor + str(round(self.std_dev, 2)) + self.resetColor + " Kg"
        else:
            ret += self.printColor + "Not trained" + self.resetColor


        return ret + "\n"


    def getTabuledSTR(self):
        """
        Devuelve un String con el nombre de la red y las metricas si las tuviera.

        ESTILO TABULADO

        Solo colorea el String si printColor y resetColor son un color.
        """
        NUM_DECIMALS = 4

        name_color = ""

        if (self.resetColor != "") and (self.printColor != ""):
            name_color = fg(242)


        ret = name_color + self.model_name + self.resetColor

        if self.trained:
            # Parametros
            ret += "\n\nnum_params: " + self.printColor + str(self.num_params) + self.resetColor
            # Metricas
            ret += "\n\tt_l: " + self.printColor + str(round(self.obtenerTrainLoss(), NUM_DECIMALS)) + self.resetColor
            ret += "\n\t\tt_err_mean: " + self.printColor + str(round(self.obtenerTrainAccuracy(), NUM_DECIMALS)) + self.resetColor + " Kg"
            ret += "\n\t\t\tval_l: " + self.printColor + str(round(self.obtenerValidationLoss(), NUM_DECIMALS)) + self.resetColor
            ret += "\n\t\t\t\tval_err_mean: " + self.printColor + str(round(self.obtenerValidationAccuracy(), NUM_DECIMALS)) + self.resetColor + " Kg"
            ret += "\n\t\t\t\t\tval_std_dev: ±" + self.printColor + str(round(self.std_dev, 2)) + self.resetColor + " Kg"
        else:
            ret += self.printColor + "Not trained" + self.resetColor


        return ret + "\n"


    def getExtendedSTR(self):
        """
        Devuelve un String con GRAN cantidad de informacion sobre la red.

        ESTILO EXTENDIDO

        Solo colorea el String si printColor y resetColor son un color.
        """
        return "[NYI]"
        NUM_DECIMALS = 2

        name_color = ""

        if (self.resetColor != "") and (self.printColor != ""):
            name_color = fg(242)


        ret = name_color + self.model_name + self.resetColor

        if self.trained:
            # Parametros
            ret += "\nnum_params: " + self.printColor + str(self.num_params) + self.resetColor
            # Metricas
            ret += "\n\tt_l: " + self.printColor + str(round(self.obtenerTrainLoss(), NUM_DECIMALS)) + self.resetColor
            ret += "\n\t\tt_acc: " + self.printColor + str(round(self.obtenerTrainAccuracy()*100, NUM_DECIMALS)) + self.resetColor + " Kg"
            ret += "\n\t\t\tval_l: " + self.printColor + str(round(self.obtenerValidationLoss(), NUM_DECIMALS)) + self.resetColor
            ret += "\n\t\t\t\tval_acc: " + self.printColor + str(round(self.obtenerValidationAccuracy()*100, NUM_DECIMALS)) + self.resetColor + " Kg"
        else:
            ret += self.printColor + "Not trained" + self.resetColor


        return ret + "\n"


    def iniciarEntrenamiento(self):
        """
        Inicia el entrenamiento del modelo durante todas las epocas.

        Es bloqueante.
        """
        # Obtenemos los data loaders
        train_loader, validation_loader = self.__obtenerDataLoader()

        # Construimos la red
        self.model = self.__construirRed(self.net_layer_struct)
        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Definimos la funcion de coste y el optimizador
        loss_fn, optimizer = self.__obtenerFuncionDeCosteYOptimizador(self.model)

        # Entrenamos el modelo (BLOQUEANTE!)
        sendMSG("Iniciando entrenamiento de la red...")
        self.__iniciarEntrenamiento(self.model, loss_fn, optimizer, train_loader, validation_loader)
        sendMSG("Entrenamiento de la red terminado.")

        # Guardamos la grafica del entrenamiento y el modelo
        self.guardarGrafica()
        self.guardarModelo()
        self.trained = True

        # Enviamos las metricas finales
        sendMSG(self.getTabuledSTR())



    def obtenerMetricasFinales(self):
        """
        Devuelve todas las metricas en un dict.

        El dict contiene listas siguiendo la siguiente estructura:

                    }
                       "loss": [],
                       "val_loss": [],
                       "accuracy": [],
                       "val_accuracy": []
                    }

        Donde cada lista contiene valores ordenados por
        epocas segun el indice.
        """
        if self.trained:
            return self.history
        else:
            sendMSG("Aun no se ha entrenado este modelo.", is_warning=True)


    def obtenerTrainLoss(self, epoch=None):
        """
        Devuelve el Train Loss de una epoca.

        Si  no se especifica ningun valor, devuelve el Train Loss definitivo
        de la red.
        """
        if self.trained:
            if epoch is None:
                return self.net_train_loss
            else:
                return self.history["loss"][-1]
        else:
            sendMSG("Aun no se ha entrenado este modelo.", is_warning=True)


    def obtenerValidationLoss(self, epoch=None):
        """
        Devuelve el Validation Loss de una epoca.

        Si  no se especifica ningun valor, devuelve el Validation Loss definitivo
        de la red.
        """
        if self.trained:
            if epoch is None:
                return self.net_val_loss
            else:
                return self.history["val_loss"][-1]
        else:
            sendMSG("Aun no se ha entrenado este modelo.", is_warning=True)


    def obtenerTrainAccuracy(self, epoch=None):
        """
        Devuelve el Train Accuracy de una epoca.

        Si  no se especifica ningun valor, devuelve el Train Accuracy definitivo
        de la red.
        """
        if self.trained:
            if epoch is None:
                return self.net_train_accuracy
            else:
                return self.history["accuracy"][-1]
        else:
            sendMSG("Aun no se ha entrenado este modelo.", is_warning=True)


    def obtenerValidationAccuracy(self, epoch=None):
        """
        Devuelve el Validation Accuracy de una epoca.

        Si  no se especifica ningun valor, devuelve el Validation Accuracy definitivo
        de la red.
        """
        if self.trained:
            if epoch is None:
                return self.net_val_accuracy
            else:
                return self.history["val_accuracy"][-1]
        else:
            sendMSG("Aun no se ha entrenado este modelo.", is_warning=True)


    def guardarGrafica(self):
        """
        Guarda la grafica en la ruta plot/ .

        Si la red tiene nan o inf en las metricas, entonces no guarda nada.
        """
        # Comprobamos si hay nan o inf en las metricas
        if self.nan_or_inf:
            return
        else:
            # Guardamos la grafica con el nombre del modelo
            self.__guardarGrafica()


    def guardarModelo(self):
        """
        Guarda el modelo de la clase en la ruta models/ .

        Si la red tiene nan o inf en las metricas, entonces no gurada nada.
        Tampoco lo hace si no se ha creado el modelo.
        """
        # Comprobamos si hay nan o inf en las metricas
        if self.nan_or_inf or (self.model is None):
            return
        else:
            # Guardamos el modelo
            self.__guardarModelo()



    def __obtenerDataLoader(self):
        """
        Funcion que crea y devuelve un par de data loaders, uno de
        entrenamiento y otro de validacion.

        Los data loaders utilizan el dataset loader propio de ClasiLamb y
        contienen la cantidad de imagenes especificada por la variable
        de clase training_percent
        """
        # Transformaciones
        img_transforms_list = [transforms.ToTensor(), transforms.Normalize((0),(2000)) ]
        img_transform = transforms.Compose(img_transforms_list)


        # dataset (normalizamos las etiquetas entre 0 y 35 para los pesos
        dataset = dynamic_dataset_loader.CLDL_r1(self.dataset_path, img_transform=img_transform, manual_label_normalize= self.manual_label_normalize)
        #sendMSG("Dataset:\n" + str(dataset))

        # Data loader de entrenamiento y validacion
        t_number = int(len(dataset) * self.training_percent)
        v_number = len(dataset) - t_number
        sendMSG("img_entrenamiento: " + str(t_number) + "   img_validacion: " + str(v_number))
        train_set, validation_set = torch.utils.data.random_split(dataset,[t_number,v_number])
        train_loader = DataLoader(dataset=train_set,
                                  shuffle=self.shuffle,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)
        validation_loader = DataLoader(dataset=validation_set,
                                       shuffle=self.shuffle,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory)

        return train_loader, validation_loader



    def __construirRed(self, net_layer_struct):
        """
        Funcion que construye y devuelve el modelo de la red neuronal en
        funcion de la estructura net_layer_struct dada por Parametros

        Para mas informacion sobre la forma que net_layer_struct
        debe tener, mirar el comentario del constructor de CL_CustomModel
        en custom_model.

        """
        model = CL_CustomModel(net_layer_struct, device)

        return model




    def __obtenerFuncionDeCosteYOptimizador(self, model):
        """
        Funcion que define y devuelve la funcion de coste y el optimizador
        """
        # Definimos la funcion de coste (la que calcula el error)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        # Definimos el optimizador que se encargara de hacer el descenso del gradiente
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        return loss_fn, optimizer




    def __checkAccuracy(self, loader, model, history=None, loss_fn=None):
        """ Funcion para evaluar el modelo con los datos que ofrezca el loader """

        model.eval()
        val_loss_list = []

        with torch.no_grad():
            for x, y in loader:

                x = x.to(device=device).float()
                y = y.to(device=device).float()

                scores = model(x)

                if (history is not None) and (loss_fn is not None):
                    val_loss_list.append(loss_fn(scores, y).item())

        if (history is not None) and (loss_fn is not None):
            val_loss = np.mean(val_loss_list)
            std_dev_loss = np.std(val_loss_list)
            history["val_loss"].append(val_loss)
            if self.manual_label_normalize is None:
                history["val_accuracy"].append(float(val_loss))
            else:
                history["val_accuracy"].append(float(val_loss * self.manual_label_normalize))


        model.train()
        if self.manual_label_normalize is None:
            return val_loss, std_dev_loss
        else:
            return (val_loss * self.manual_label_normalize), (std_dev_loss * self.manual_label_normalize)




    def __acumAndGetMeanLoss(self, new_loss_number):
        """
        Funcion que acumula el error y devuelve la media y desviacion
        tipica de todos los errores acumulados
        """
        self.loss_hist = np.append(self.loss_hist, new_loss_number)

        return float(np.mean(self.loss_hist)), float(np.std(self.loss_hist))




    def __resetAcumLoss(self):
        """ Funcion que resetea el historial de metricas loss. """
        self.loss_hist = np.array([])




    def __iniciarEntrenamiento(self, model, loss_fn, optimizer, train_loader, validation_loader):
        """
        Funcion privada que inicia el entrenamiento.

        La funcion es bloqueante y realiza el entrenamiento completo del
        modelo a traves de todas las epocas.
        Una vez el entrenamiento termina, se almacenan las metricas finales en
        la variable de clase history.
        """

        # Todo lo que envueleve a donex es para poder ver los datos del dataset.
        # Poniendolo a True se quita dicha funcionalidad
        donex = False

        # Historial de metricas
        history = {
                   "loss": [],
                   "val_loss": [],
                   "accuracy": [],
                   "val_accuracy": []
                   }

        # Ponemos el modelo en modo entrenamiento
        model.train()

        bar_manager = enlighten.get_manager()
        epochs_bar = bar_manager.counter(total=self.epochs, desc="Epochs:  ", unit='Epochs', position=2, leave=True, color=(150,255,0))


        # Entrenamos!!            ======================  loop  =======================
        for epoch in range(self.epochs):

            ent_loss_list = []
            num_correct = 0
            num_samples = 0

            train_bar = bar_manager.counter(total=len(train_loader), desc="Training:  ", unit='img', position=1, leave=False, color=(50,150,0))
            for imgs, labels in train_loader:
                # Preparamso las imagenes
                imgs = imgs.to(device)
                labels = labels.to(device)
                if not donex:
                    print(self.C + "--------------------------- Datos de los Tensores del Dataset ---------------------------\n\n")
                    #print(torch.unique(imgs))
                    #print(imgs)
                    print("dimensiones:  ", imgs.size())
                    print("dtype:  ", imgs.dtype , "\n\n")

                    print("Label:    ", labels, "     dimensiones:  ", labels.size(), "    dtype:  ", labels.dtype , "\n\n" + self.B)
                    donex = True

                # Sacamos las predicciones
                outputs = model(imgs)
                print("------  outputs: " + str(outputs))
                print("------  labels: "  +  str(labels))

                # Obtenemos el error
                loss = loss_fn(outputs, labels)
                ent_loss_list.append(loss.item())
                print("------  loss: "  +  str(loss.item()))


                # Back-propagation y entrenamiento
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Tick de la barra de entrenamiento
                loss_mean, loss_std = self.__acumAndGetMeanLoss(loss.item())
                train_bar.desc = "Trainig:  loss= " + str(round(loss_mean,4)) + "  std_dev= " + str(round(loss_std,2)) + " "
                train_bar.update()


            # Guardamos las metricas de la epoca
            history["loss"].append(np.mean(ent_loss_list))
            if self.manual_label_normalize is None:
                history["accuracy"].append(np.mean(ent_loss_list))
            else:
                history["accuracy"].append(np.mean(ent_loss_list) * self.manual_label_normalize)


            # Comprobamos si hay NaN o Inf en las metricas
            if isnan(history["loss"][-1]) or isinf(history["loss"][-1]):
                # Guardamos las metricas y paramos de entrenar, pues seria inutil continuar
                history["val_loss"].append(float("NaN"))
                history["val_accuracy"].append(float("NaN"))
                self.history = history
                bar_manager.remove(train_bar)
                self.nan_or_inf = True
                # Guardamos las stats de NaN
                self.net_train_loss = float("NaN")
                self.net_train_accuracy = float("NaN")
                self.net_val_loss = float("NaN")
                self.net_val_accuracy = float("NaN")
                self.std_dev = float("NaN")
                sendMSG("La red contiene NaN", is_warning=True)
                break


            # Borramos la barra de entrenamiento
            bar_manager.remove(train_bar)
            self.__resetAcumLoss()

            # Tick de la barra de epocas
            mean_loss, std_dev_loss = self.__checkAccuracy(validation_loader, model, history, torch.nn.L1Loss())
            prefix_epochs_bar = "Epochs:  val_acc= "+str(round(mean_loss, 4))+" Kg "
            epochs_bar.desc = prefix_epochs_bar
            epochs_bar.update()


            # Early stop. Guardamos la mejor epoca hasta ahora
            # Si esta epoca es la mejor:
            test_loss = history["val_loss"][-1]
            test_acc  = history["val_accuracy"][-1]
            if test_loss < self.best_test_loss:
                # Guardamos su modelo temporalmente
                self.current_best_model = deepcopy(model)

                # Guardamos sus metricas
                self.net_train_loss = history["loss"][-1]
                self.net_train_accuracy = history["accuracy"][-1]
                self.net_val_loss = test_loss
                self.net_val_accuracy = test_acc

                # Actualizamos el mejor loss
                self.best_test_loss = test_loss
                self.std_dev = std_dev_loss


            # Mostramos las metricas
            colors = ["#ff6163", "#ff964f", "#20c073", "#b1ff65"]
            names = ["loss", "val_loss", "err_mean", "val_err_mean"]
            print("e " + str(epoch) + ":\t ", end="")
            for i, key in enumerate(history):
                print(names[i] + ": " + fg(colors[i]) + str(round(history[key][epoch], 4)) + self.B + "  ", end="")
            print()

            # Guardamos el historial de metricas
            self.history = history

        # Destruimos las barras
        bar_manager.remove(epochs_bar)

        # Reestablecemos el modelo al modelo de la mejor epoca
        self.model = self.current_best_model




    def __guardarModelo(self):
        """ Funcion privada que guarda el modelo dado. """
        # Guardamos el modelo final
        if not os.path.exists(self.saved_models_path):
            os.makedirs(self.saved_models_path)

        model_name_and_path = os.path.join(self.saved_models_path, self.model_name)
        torch.save(self.model, model_name_and_path)


    def __guardarGrafica(self):
        """ Funcion privada que guarda la grafica con el historial de metricas. """

        # Gestionamos la grafica
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

        plot_name_and_path = os.path.join(self.plots_path, self.model_name)
        show_plot_and_save(self.history,regression=True, just_save=True, save_name=plot_name_and_path)


    def loadFromJson(self, json_data):
        """
        Carga los valores de los atributos desde un json

        Soporta redes entrenadas y no entrenadas.
        """

        if json_data["trained"] == True:
            # Cargamos los atributos y las metricas
            # Cargamos lo inprescindible
            self.net_layer_struct = json_data["net_layer_struct"]
            self.hyperparams = json_data["hyperparams"]

            # Cargamos las metricas
            self.history = json_data["history"]

            # Cargamos la info extra
            self.trained = json_data["trained"]
            self.num_params = json_data["num_params"]
            self.nan_or_inf = json_data["nan_or_inf"]
            self.creation_date = json_data["creation_date"]
            self.last_modification_date = json_data["last_modification_date"]
            self.net_train_loss = json_data["net_train_loss"]
            self.net_train_accuracy = json_data["net_train_accuracy"]
            self.net_val_loss = json_data["net_val_loss"]
            self.net_val_accuracy = json_data["net_val_accuracy"]
            # Propiedades nuevas a guardar
            try:
                self.std_dev = json_data["std_dev"]
            except:
                self.std_dev = -1

        else:
            # Cargamos los atributos para el entrenamiento
            # Cargamos lo imprescindible
            self.net_layer_struct = json_data["net_layer_struct"]
            self.hyperparams = json_data["hyperparams"]

            # Cargamos info extra
            self.trained = json_data["trained"]
            self.creation_date = json_data["creation_date"]
            self.last_modification_date = json_data["last_modification_date"]
