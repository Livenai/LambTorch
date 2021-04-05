import os, enlighten
import numpy as np
from colored import fg

import torch
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
device = ("cuda" if torch.cuda.is_available() else "cpu")

import dynamic_dataset_loader
from plot import show_plot_and_save
from custom_model import CL_CustomModel


# Parametros
learning_rate = 0.00001
batch_size = 1
epochs = 50

training_percent = 0.9 # 90% de imagenes para entrenamiento
shuffle = True
pin_memory = True
num_workers = 1
model_name = "CL21_M1_50ep"


# Colores
B = fg(15)
C = fg(154)


# SEED
torch.manual_seed(42)

# PATH
parent_folder = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(parent_folder, "clasiLamb_2-1_CUS")
saved_models_path = os.path.join(parent_folder, "saved_models")
plots_path = os.path.join(parent_folder, "plots")
