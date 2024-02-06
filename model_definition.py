import numpy as np
import torch


def normalization(x):
    return (x - -100)/(0 - -100)

beacons_cols = ['F0B1', 'F0B2', 'F0B3', 'F0B4', 'F0B5', 'F0B6', 'F0B7', 'T01B1', 'F1B1',
                'F1B2', 'F1B3', 'T12B1', 'F4B1', 'F4B2',
                'T23B1', 'T34B1', 'T45B1', 'F2B1', 'F2B2',
                'F2B3', 'F3B1', 'F3B2', 'F5B1', 'F5B2']

class Encoder(torch.nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim, window):
        super().__init__()

        self.encoder_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 16, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.ReLU(True)
        )

        self.flatten = torch.nn.Flatten(start_dim=1)

        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(3 * window * 32, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(torch.nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim, window):
        super().__init__()
        self.decoder_lin = torch.nn.Sequential(
            torch.nn.Linear(encoded_space_dim, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 3 * window * 32),
            torch.nn.ReLU(True)
        )

        self.unflatten = torch.nn.Unflatten(dim=1,
        unflattened_size=(32, window, 3))

        self.decoder_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 1, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class MultiOutputRegressionHead(torch.nn.Module):
    def __init__(self, encoded_space_dim=9):
        super(MultiOutputRegressionHead, self).__init__()
        self.hid1 = torch.nn.Linear(encoded_space_dim, 32) # nb in, nb out ?
        self.hid2 = torch.nn.Linear(32, 32)
        self.outp = torch.nn.Linear(32, 2)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.outp.weight)
        torch.nn.init.zeros_(self.outp.bias)

    def forward(self, x, nb_layers=1):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = self.outp(z)

        return z
