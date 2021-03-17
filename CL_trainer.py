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

import torch
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
device = ("cuda" if torch.cuda.is_available() else "cpu")

import dynamic_dataset_loader
from plot import show_plot_and_save
from custom_model import CL_CustomModel



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
        self.learning_rate = hyperparams["learning_rate"]
        self.batch_size = hyperparams["batch_size"]
        self.epochs = hyperparams["epochs"]
        self.training_percent = hyperparams["training_percent"]
        self.model_name = hyperparams["model_name"]

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
        self.dataset_path = os.path.join(self.parent_folder, "clasiLamb_2-1_CUS")
        self.saved_models_path = os.path.join(self.parent_folder, "saved_models")
        self.plots_path = os.path.join(self.parent_folder, "plots")

        # Parametros extra
        self.loss_hist = np.array([])
        self.history = {}
        self.trained = False
        self.num_params = None
        self.nan_or_inf = False
        self.color = ""


    def __str__(self):
        """
        Imprime el nombre de la red y las metricas si las tuviera.
        """
        NUM_DECIMALS = 2

        ret = fg(242) + self.model_name + fg(15)
        ret += "  |\n\t|  "

        if self.trained:
            # Parametros
            ret += "num_params: " + self.color + str(self.num_params) + fg(15)
            # Metricas
            ret += "  t_l: " + self.color + str(round(self.obtenerTrainLoss(), NUM_DECIMALS)) + fg(15)
            ret += "  t_acc: " + self.color + str(round(self.obtenerTrainAccuracy()*100, NUM_DECIMALS)) + fg(15) + "%"
            ret += "  val_l: " + self.color + str(round(self.obtenerValidationLoss(), NUM_DECIMALS)) + fg(15)
            ret += "  val_acc: " + self.color + str(round(self.obtenerValidationAccuracy()*100, NUM_DECIMALS)) + fg(15) + "%"
        else:
            ret += self.color + "Not trained" + fg(15)


        return ret + "\n"



    def iniciarEntrenamiento(self):
        """
        Inicia el entrenamiento del modelo durante todas las epocas.

        Es bloqueante.
        """
        # Obtenemos los data loaders
        train_loader, validation_loader = self.__obtenerDataLoader()

        # Construimos la red
        model = self.__construirRed(self.net_layer_struct, prints=True)
        self.num_params = sum(p.numel() for p in model.parameters())

        # Definimos la funcion de coste y el optimizador
        loss_fn, optimizer = self.__obtenerFuncionDeCosteYOptimizador(model)

        # Entrenamos el modelo (BLOQUEANTE!)
        self.__iniciarEntrenamiento(model, loss_fn, optimizer, train_loader, validation_loader)

        self.trained = True



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
            print(fg(1) + "[!] Aun no se ha entrenado este modelo." + fg(15))

    def obtenerTrainLoss(self):
        """ Devuelve el Train Loss final. """
        if self.trained:
            return self.history["loss"][-1]
        else:
            print(fg(1) + "[!] Aun no se ha entrenado este modelo." + fg(15))


    def obtenerValidationLoss(self):
        """ Devuelve el Validation Loss final. """
        if self.trained:
            return self.history["val_loss"][-1]
        else:
            print(fg(1) + "[!] Aun no se ha entrenado este modelo." + fg(15))

    def obtenerTrainAccuracy(self):
        """ Devuelve el Train Accuracy final. """
        if self.trained:
            return self.history["accuracy"][-1]
        else:
            print(fg(1) + "[!] Aun no se ha entrenado este modelo." + fg(15))


    def obtenerValidationAccuracy(self):
        """ Devuelve el Validation Accuracy final. """
        if self.trained:
            return self.history["val_accuracy"][-1]
        else:
            print(fg(1) + "[!] Aun no se ha entrenado este modelo." + fg(15))


    def guardarGrafica(self):
        """
        Guarda la grafica en la ruta dada.

        Si no se especifica ruta, la guarda en la carpeta por defecto (/plot)
        con el nombre del modelo y las epocas.
        """
        pass




    def __obtenerDataLoader(self):
        """
        Funcion que crea y devuelve un par de data loaders, uno de
        entrenamiento y otro de validacion.

        Los data loaders utilizan el dataset loader propio de ClasiLamb y
        contienen la cantidad de imagenes especificada por la variable
        de clase training_percent
        """
        # Transformaciones
        transforms_list = [transforms.ToTensor(), transforms.Normalize((0),(2000)) ]
        transform = transforms.Compose(transforms_list)

        # dataset
        dataset = dynamic_dataset_loader.CLDL_b1(self.dataset_path, transform=transform)
        print("Dataset:\n",str(dataset))

        # Data loader de entrenamiento y validacion
        t_number = int(len(dataset) * self.training_percent)
        v_number = len(dataset) - t_number
        print("t_number: ", t_number, "  v_number: ", v_number)
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



    def __construirRed(self, net_layer_struct, prints=False):
        """
        Funcion que construye y devuelve el modelo de la red neuronal en
        funcion de la estructura net_layer_struct dada por Parametros

        Para mas informacion sobre la forma que net_layer_struct
        debe tener, mirar el comentario del constructor de CL_CustomModel
        en custom_model.
        """


        model = CL_CustomModel(net_layer_struct, device)

        if prints:
            print("===================  Modelo  ===================")
            print(model)
            print("\n")
            summary(model, (1,480,640))
            print("================================================")

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

        num_correct = 0
        num_samples = 0
        model.eval()
        val_loss_list = []

        with torch.no_grad():
            for x, y in loader:

                x = x.to(device=device).float()
                y = y.to(device=device).float()

                scores = model(x)
                predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                if (history is not None) and (loss_fn is not None):
                    val_loss_list.append(loss_fn(scores, y).item())

        if (history is not None) and (loss_fn is not None):
            val_loss = np.mean(val_loss_list)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(float(num_correct)/float(num_samples))



        #print(f"- Val_acc test:  {num_correct} / {num_samples} imgs correctamente predichas ({float(num_correct)/float(num_samples)*100:.2f} %)")
        model.train()

        return f"{float(num_correct)/float(num_samples)*100:.2f}"




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
                predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in outputs]).to(device)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)


                # Obtenemos el error
                loss = loss_fn(outputs, labels)
                ent_loss_list.append(loss.item())

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
            history["accuracy"].append(float(num_correct)/float(num_samples))

            # Comprobamos si hay NaN o Inf en las metricas
            if isnan(history["loss"][-1]) or isinf(history["loss"][-1]):
                # Guardamos las metricas y paramos de entrenar, pues seria inutil continuar
                history["val_loss"].append(float("NaN"))
                history["val_accuracy"].append(float("NaN"))
                self.history = history
                bar_manager.remove(train_bar)
                self.nan_or_inf = True
                break



            # Borramos la barra de entrenamiento
            bar_manager.remove(train_bar)
            self.__resetAcumLoss()

            # Tick de la barra de epocas
            prefix_epochs_bar = "Epochs:  val_acc= "+str(self.__checkAccuracy(validation_loader, model, history, loss_fn))+"% "
            epochs_bar.desc = prefix_epochs_bar
            epochs_bar.update()

            # Mostramos las metricas
            colors = ["#ff6163", "#ff964f", "#20c073", "#b1ff65"]
            print("e " + str(epoch) + ":\t ", end="")
            for i, key in enumerate(history):
                print(str(key) + ": " + fg(colors[i]) + str(round(history[key][epoch], 4)) + self.B + "  ", end="")
            print()

            # Guardamos el historial de metricas
            self.history = history

        # Destruimos las barras
        bar_manager.remove(epochs_bar)




    def __guardarModelo(self, model):
        """ Funcion privada que guarda el modelo dado. """
        # Guardamos el modelo final
        if not os.path.exists(self.saved_models_path):
            os.makedirs(self.saved_models_path)

        model_name_and_path = os.path.join(self.saved_models_path, self.model_name)
        torch.save(model, model_name_and_path)


    def __guardarGrafica(self):
        """ Funcion que guarda la grafica con el historial de metricas. """

        # Gestionamos la grafica
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

        plot_name_and_path = os.path.join(self.plots_path, self.model_name)
        show_plot_and_save(self.history, just_save=True, save_name=plot_name_and_path)
