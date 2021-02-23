import torch
from torch import nn


class M1_v1(nn.Module):
    def __init__(self, img_channels):
        """
        Ctor.

        Hay que mover el modelo a la gpu
        """
        super(M1_v1, self).__init__()

        self.cnn_layers = nn.Sequential(
            # in_channels debe ser los canales de la imagen (3 en rgb o 1 en grayscale)
            nn.Conv2d(in_channels=img_channels, out_channels=20, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.cnn_layers = self.cnn_layers.cuda().float()

        self.linear_layers = nn.Sequential(
            nn.Linear(200*28*38,500),nn.Sigmoid(), # in_features debe ser out_channels anterior * dimensiones de la imagen de salida (pues hace de flatten)
            nn.Linear(500,200),nn.Sigmoid(),
            nn.Linear(200,50),nn.Sigmoid(),
            nn.Linear(50,1),nn.Sigmoid()
        )

        self.linear_layers = self.linear_layers.cuda().float()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
