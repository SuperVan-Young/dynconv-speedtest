import sys

import dynconv.pytorch
import dynconv.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F

conv_sampler = utils.ConvSampler()
weight = conv_sampler.gen_weight()
bias = conv_sampler.gen_bias()
ifmap = conv_sampler.gen_input()
mask = conv_sampler.gen_mask()
print(mask)

ofmap = None
with torch.no_grad():
    dynconv_torch = dynconv.pytorch.DynconvTorch()
    with utils.Timer(dynconv_torch) as t:
        ofmap = dynconv_torch(weight, ifmap, mask, bias)
    print(dynconv_torch.elapsed_time)
