import CL_trainer
from CL_trainer import CL_Trainer
import hyperparams_generator
from hyperparams_generator import getRandomHyperParamsV1, printDict, printNetLayers





# Obtenemos los hiperparametros y la morfologia de la red
hyperparams, net_layers = getRandomHyperParamsV1()


printDict(hyperparams)
print("-----------------------------------------------------------")
printNetLayers(net_layers)




train01 = CL_Trainer(hyperparams, net_layers)

train01.iniciarEntrenamiento()
