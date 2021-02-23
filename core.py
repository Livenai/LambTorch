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



# Transformaciones
transforms_list = [transforms.ToTensor(), transforms.Normalize((0),(2000)) ]
transform = transforms.Compose(transforms_list)


# dataset
dataset = dynamic_dataset_loader.CLDL_b1(dataset_path, transform=transform)

t_number = int(len(dataset) * training_percent)
v_number = len(dataset) - t_number
print("t_number: ", t_number, "  v_number: ", v_number)
train_set, validation_set = torch.utils.data.random_split(dataset,[t_number,v_number])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)


print("Dataset:\n",str(dataset))



# Construimos la red
from M1 import M1_v1
model = M1_v1(img_channels = 1)
print("===================  Modelo  ===================")
print(model)
print("\n")
summary(model, (1,480,640))
print("================================================")


# Definimos la funcion de coste (la que calcula el error)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Definimos el optimizador que se encargara de hacer el descenso del gradiente
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





def check_accuracy(loader, model, history=None, loss_fn=None):
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


loss_hist = np.array([])
def acum_and_get_mean_loss(new_loss_number):
    """
    Funcion que acumula el error y devuelve la media y desviacion
    tipica de todos los errores acumulados
    """
    global loss_hist
    loss_hist = np.append(loss_hist, new_loss_number)

    return float(np.mean(loss_hist)), float(np.std(loss_hist))



def reset_acum_loss():
    global loss_hist
    loss_hist = np.array([])





torch.set_printoptions(edgeitems=3)

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
epochs_bar = bar_manager.counter(total=epochs, desc="Epochs:  ", unit='Epochs', position=2, leave=True, color=(150,255,0))


# Entrenamos!!            ======================  loop  =======================
for epoch in range(epochs):

    ent_loss_list = []
    num_correct = 0
    num_samples = 0

    train_bar = bar_manager.counter(total=len(train_loader), desc="Training:  ", unit='img', position=0, leave=False, color=(50,150,0))
    for imgs, labels in train_loader:
        # Preparamso las imagenes
        imgs = imgs.to(device)
        labels = labels.to(device)
        if not donex:
            print(C + "--------------------------- Datos de los Tensores del Dataset ---------------------------\n\n")
            #print(torch.unique(imgs))
            #print(imgs)
            print("dimensiones:  ", imgs.size())
            print("dtype:  ", imgs.dtype , "\n\n")

            print("Label:    ", labels, "     dimensiones:  ", labels.size(), "    dtype:  ", labels.dtype , "\n\n" + B)
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
        loss_mean, loss_std = acum_and_get_mean_loss(loss.item())
        train_bar.desc = "Trainig:  loss= " + str(round(loss_mean,4)) + "  std_dev= " + str(round(loss_std,2)) + " "
        train_bar.update()


    # Guardamos las metricas de la epoca
    history["loss"].append(np.mean(ent_loss_list))
    history["accuracy"].append(float(num_correct)/float(num_samples))


    # Borramos la barra de entrenamiento
    bar_manager.remove(train_bar)
    reset_acum_loss()

    # Tick de la barra de epocas
    prefix_epochs_bar = "Epochs:  val_acc= "+str(check_accuracy(validation_loader, model, history, loss_fn))+"% "
    epochs_bar.desc = prefix_epochs_bar
    epochs_bar.update()

    # Mostramos las metricas
    colors = ["#ff6163", "#ff964f", "#20c073", "#b1ff65"]
    print("e " + str(epoch) + ":\t ", end="")
    for i, key in enumerate(history):
        print(str(key) + ": " + fg(colors[i]) + str(round(history[key][epoch], 4)) + B + "  ", end="")
    print()



# Guardamos el modelo final
if not os.path.exists(saved_models_path):
    os.makedirs(saved_models_path)

model_name_and_path = os.path.join(saved_models_path, model_name)
torch.save(model, model_name_and_path)

# Gestionamos la grafica
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

plot_name_and_path = os.path.join(plots_path, model_name)
show_plot_and_save(history, just_save=True, save_name=plot_name_and_path)
