import CL_trainer
from CL_trainer import CL_Trainer



# Probamos el entrenamiento estandar con una instancia:
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

hyperparams = {
        "learning_rate": 0.00001,
        "batch_size": 1,
        "epochs": 10,
        "training_percent": 0.9,
        "model_name": "prueba01"
}




train01 = CL_Trainer(hyperparams, net_layers)

train01.iniciarEntrenamiento()
