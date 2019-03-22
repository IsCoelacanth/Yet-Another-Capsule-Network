import torch
import torch.nn.functional as func
import torch.nn as nn

from Capsule import CapsuleLayer
from exp_decoder import Decoder


class CapsuleNetwork(nn.Module):

    def __init__(self):
        super(CapsuleNetwork, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.p_caps = CapsuleLayer(num_caps=8, num_routes=-1, in_channels=256, out_channels=32,
                                   k_size=9, stride=2)
        self.d_caps = CapsuleLayer(num_caps=10, num_routes=32*28*28, in_channels=8, out_channels=16)

        self.decoder = Decoder()

    def forward(self, x, y=None):
        x = func.relu(self.initial_conv(x))
        # print(x.shape)
        x = self.p_caps(x)
        # print(x.shape)
        x = self.d_caps(x).squeeze().transpose(0,1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = func.softmax(classes, dim=-1)

        if y is not None:
            _, max_index = classes.max(dim=1)
            y = torch.eye(10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 requires_grad = True).index_select(dim=0, index=max_index.data)
        reconst = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconst


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconst = nn.MSELoss(size_average=False)

    def forward(self, img, label, classes, reconst):
        label = torch.eye(10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                  requires_grad=True).index_select(dim=0, index=label.data)
        # print(classes.size(), label.size())
        left = func.relu(0.9-classes) ** 2
        right = func.relu(classes - 0.1) ** 2

        margin = label * left + 0.5 * (1-label) * right
        margin = margin.sum()

        recon = self.reconst(img, reconst)
        return (margin + 0.0005 * recon) / img.size(0)

