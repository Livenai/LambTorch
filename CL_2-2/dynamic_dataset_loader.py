import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset



"""          INFO
Para crear un dataset loader solo hay que crear una clase que herede
de torch.utils.data.Dataset e implementar dos metodos:

    __len__()        - El cual devolvera la longitud total del dataset

    __getitem__(id)  - El cual servira para indexar y obtener elementos

"""





class CLDL_r1(Dataset):
    """
    ClasiLamb Dataset Loader release 1

    Cargador de datasets de imagenes desde disco para el entrenamiento de
    pesadas cantidades de datos.
    """

    def __init__(self, dataset_path_, img_transform=None, label_transform=None, manual_label_normalize = None):
        """
        Ctor. La clase necesita la ruta a la carpeta donde se encuentra el
        dataset ya preparado.

        Esta preparacion consta de las imagenes numeradas y procesadas y de un
        archivo .npy con las respuestas ordenadas respecto de los nombres de
        las imagenes.

        Esta funcion carga las respuestas (labels) para mantenerlas en
        y evitar cargarlas de nuevo mas adelante.

        manual_label_normalize - Normaliza las etiquetas entre 0 y el numero
        dado por esta variable. Si vale None, no normaliza.
        """
        self.dataset_path = dataset_path_
        self.labels = np.load(os.path.join(self.dataset_path, "labels.npy"))
        self.num_elements = self.labels.size
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.manual_label_normalize = manual_label_normalize



    def __len__(self):
        """ Devuelve la longitud del dataset """
        return int(self.num_elements)


    def __getitem__(self, idx):
        """
        Devuelve el elemento dataset[idx]

        El item devuelto es una tupla con dos tensores, el primero de datos
        y el segundo de labels.
        """

        # Comprobamos el rango
        if idx < 0 or idx >= self.num_elements:
            raise Exception("El idx recibido esta fuera de rango [0, " + str(self.num_elements-1) + "]:  idx == " + str(idx))

        # Cargamos la imagen con nombre igual al idx
        img_path = os.path.join(self.dataset_path, str(idx)+".png")
        img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32) # OJO AL astype()
        #print(np.unique(img))


        # Comprobamos si la imagen ha sido cargada correctamente
        if len(img) == 0:
            raise Exception("Error al cargar la imagen:     len(img) == 0")


        # Obtenemos la respuesta con el mismo idx
        true_y = self.labels[idx].astype(np.float32)

        # Aplicamos las transformaciones a las imagenes
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            true_y = self.label_transform(true_y)


        # Acciones manuales
        if self.manual_label_normalize is not None:
            # Dividimos entre  manual_label_normalize  para normalizar (damos por hecho que la media es 0)
            true_y = true_y / np.float32(self.manual_label_normalize)



        # Muestra a devolver
        #print("------------------img:",img,"    label:",true_y, "          datasize: ",img.shape)
        return (img, true_y)


    def fullname(self, o):
        """
        o.__module__ + "." + o.__class__.__qualname__ is an example in
        this context of H.L. Mencken's "neat, plausible, and wrong."
        Python makes no guarantees as to whether the __module__ special
        attribute is defined, so we take a more circumspect approach.
        Alas, the module name is explicitly excluded from __qualname__
        in Python 3.
        """

        module = o.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return o.__class__.__name__  # Avoid reporting __builtin__
        else:
            return module + '.' + o.__class__.__name__



    def __str__(self):
        """ Devuelve informacion del dataset """
        msg =  "DataLoader " + str(type(self).__name__)
        msg += "  |  " + str(len(self)) + " samples of type "
        s = self[0]
        msg += str(self.fullname(s[0])) + " and " + str(self.fullname(s[1]))

        return msg
