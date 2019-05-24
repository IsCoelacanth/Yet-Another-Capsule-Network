import torch
import torch.nn.functional as func
import torch.nn as nn

from Capsule import CapsuleLayer
from exp_decoder import Decoder
import math

def conv_size(shape, k = 9, s = 1, p = False):
    H, W = shape
    if p:
        pad = (k-1)//2
    else:
        pad = 0

    Ho = math.floor(((H + 2*pad - (k - 1) - 1)/s) + 1)
    Wo = math.floor(((W + 2*pad - (k - 1) - 1)/s) + 1)

    return Ho, Wo


class CapsuleNetwork(nn.Module):
    def __init__(self, img_size, ic_channels, num_pcaps, num_classes, num_coc, num_doc, mode='mono', use_padding=False):
        super(CapsuleNetwork, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels=1 if mode=='mono' else 3, out_channels=ic_channels, kernel_size=9, stride=1)
        Ho, Wo = conv_size(img_size, k=9, s=1, p=False)

        self.p_caps = CapsuleLayer(num_caps=num_pcaps, num_routes=-1, in_channels=ic_channels, out_channels=num_coc,
                                   k_size=9, stride=2)
        Ho, Wo = conv_size((Ho, Wo), k=9, s=2, p=use_padding)

        self.d_caps = CapsuleLayer(num_caps=num_classes, num_routes=num_coc*Ho*Wo, in_channels=num_pcaps, out_channels=num_doc)

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
        # print(img.size(), reconst.size())
        left = func.relu(0.9-classes) ** 2
        right = func.relu(classes - 0.1) ** 2

        margin = label * left + 0.5 * (1-label) * right
        margin = margin.sum()

        recon = self.reconst(img, reconst)
        return (margin + 0.0005 * recon) / img.size(0)

