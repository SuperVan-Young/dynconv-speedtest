import torch
import torch.nn as nn
import torch.nn.functional as F

class DynconvTorch():

    def __call__(self, weight, ifmap, mask, bias):
        x = torch.mul(ifmap, mask)
        x = F.conv2d(x, weight, bias)
        return x
