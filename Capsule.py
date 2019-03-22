import torch
import torch.nn.functional as func
import torch.nn as nn


class CapsuleLayer(nn.Module):
    """
    Basic Capsule layer: when num_routes is >0 it acts as the dynamic routing Digit Cap layer, other wise as a primary
    capsule layer
    """
    def __init__(self, num_caps, num_routes, in_channels, out_channels, k_size=None, stride=None, num_rounds=3):
        super(CapsuleLayer, self).__init__()

        self.num_routes = num_routes
        self.num_rounds = num_rounds
        self.num_caps = num_caps

        if num_routes != -1:
            self.W = nn.Parameter(torch.randn(num_caps, num_routes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=(k_size-1)//2)
                 for _ in range(num_caps)]
            )

    @staticmethod
    def squash(x, dim=-1):
        s_norm = (x**2).sum(dim=dim, keepdim=True)
        scaled = s_norm / (1 + s_norm)
        return scaled * x / torch.sqrt(s_norm)

    def forward(self, x):

        if self.num_routes != -1:
            # print(x.shape, self.W.shape)
            priors = x[None, :, :, None, :] @ self.W[:, None, :, :, :]
            logits = torch.zeros(*priors.size(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 requires_grad = True)
            for i in range(self.num_rounds):
                probs = func.softmax(logits, dim=2)
                outps = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_rounds - 1:
                    del_logits = (priors * outps).sum(dim=-1, keepdim=True)
                    logits = logits + del_logits
        else:
            outps = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
            outps = torch.cat(outps, dim=-1)
            outps = self.squash(outps)
        return outps

