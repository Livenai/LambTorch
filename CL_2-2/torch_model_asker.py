import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms




class ClasiLamb_2_1_ModelAsker():
    """
    Esta clase carga el modelo y permite utilizarlo para hacer predicciones
    """

    def __init__(self):
        self.model = None


    def loadModel(self, model_path):
        """ Carga el modelo del archivo """
        model_name = self.__get_just_filename(model_path)

        if "CL21" in model_name:
            self.model = torch.load(model_path)
            self.model.eval()
        else:
            print("[!] Modelo no cargado. El modelo \"" + model_name + "\" no es un modelo CL21 valido. (ClasiLamb 2-1)")

    def ask(self, input : np.ndarray):
        """
        Realiza una prediccion usando el input sobre el modelo.
        El input debe ser un np.ndarray (imagen de opencv)
        Si cuda esta disponible, intenta usar la gpu.
        Transforma en tensores y adapta los datos pacordes a las necesidades de la red

        Return: (resultado, raw_output)
        """
        # Comprobamos si el modelo esta cargado
        if self.model is None:
            print("[!] El modelo no ha sido previamente cargado. Carga el modelo antes de usarlo.")

        # Seleccionamos el dispositivo
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Adaptamos los datos
        transforms_list = [transforms.ToTensor(), transforms.Normalize((0),(2000)) ]
        transform = transforms.Compose(transforms_list)

        input_as_tensor = transform(input.astype(np.float32)).to(device)

        # recuerda que la _ es para que se aplique al tensor actual
        input_as_tensor.unsqueeze_(0)



        # Usamos los datos para obtener la prediccion
        pred = self.model(input_as_tensor)

        # adaptamos y devolvemos la prediciion
        pred = pred.item()
        if pred >= 0.5:
            return 1.0, pred
        else:
            return 0.0, pred


    def __get_just_filename(self, path):
        """Devuelve el nombre del archivo sin extension."""
        base = os.path.basename(path)
        return os.path.splitext(base)[0]


















class ClasiLamb_2_2_ModelAsker():
    """
    Esta clase carga el modelo y permite utilizarlo para hacer predicciones
    """

    def __init__(self):
        self.model = None


    def loadModel(self, model_path):
        """ Carga el modelo del archivo """
        model_name = self.__get_just_filename(model_path)

        if "CL22" in model_name:
            self.model = torch.load(model_path)
            self.model.eval()
        else:
            print("[!] Modelo no cargado. El modelo \"" + model_name + "\" no es un modelo CL22 valido. (ClasiLamb 2-2)")

    def ask(self, input : np.ndarray):
        """
        Realiza una prediccion usando el input sobre el modelo.
        El input debe ser un np.ndarray (imagen de opencv)
        Si cuda esta disponible, intenta usar la gpu.
        Transforma en tensores y adapta los datos pacordes a las necesidades de la red

        Return: raw_output (float)
        """
        # Comprobamos si el modelo esta cargado
        if self.model is None:
            print("[!] El modelo no ha sido previamente cargado. Carga el modelo antes de usarlo.")

        # Seleccionamos el dispositivo
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Adaptamos los datos
        transforms_list = [transforms.ToTensor(), transforms.Normalize((0),(2000)) ]
        transform = transforms.Compose(transforms_list)

        input_as_tensor = transform(input.astype(np.float32)).to(device)

        # recuerda que la _ es para que se aplique al tensor actual
        input_as_tensor.unsqueeze_(0)



        # Usamos los datos para obtener la prediccion
        pred = self.model(input_as_tensor)

        # adaptamos y devolvemos la prediciion
        return pred.item()



    def __get_just_filename(self, path):
        """Devuelve el nombre del archivo sin extension."""
        base = os.path.basename(path)
        return os.path.splitext(base)[0]
