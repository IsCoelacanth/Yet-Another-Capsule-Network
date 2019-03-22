import torch
import torch.nn.functional as func
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(16 * 10, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1536) # 1/8
        self.c1 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1) #1/4
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) #1/2
        self.c3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1) #1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.view(-1, 24, 8, 8)
        x = func.interpolate(x, scale_factor=2)
        x = self.relu(self.c1(x))
        x = func.interpolate(x, scale_factor=2)
        x = self.relu(self.c2(x))
        x = func.interpolate(x, scale_factor=2)
        x = torch.sigmoid(self.c3(x))
        return x
