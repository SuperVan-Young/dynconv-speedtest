from curses.panel import bottom_panel
from logging import warning
import os
from tkinter.tix import Tree

import sparseconv.paper
import sparseconv.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE == "cpu":
    warning("Torch is using CPU, the measurement may be incorrect.")
SPARSITY = 0.70
REPETITION = 100

# ih/iw, cin, expand_ratio, cout, stride
# parameters from mobilenet-v2

# testcase with identity shortcut
testcases = [
    (56, 24, 6, 24, 1),
    (28, 32, 6, 32, 1),
    (14, 64, 6, 64, 1),
    (14, 96, 6, 96, 1),
    (7, 160, 6, 160, 1),
]

# before testing, need to warm up GPU
warmup = torch.randn(3, 4).to(DEVICE)
warmup = F.relu(warmup)
del warmup

for testcase in testcases:
    width, cin, expand_ratio, cout, stride = testcase
    sampler = utils.ConvSampler(
        channel_in=cin, channel_out=cout, sparsity=SPARSITY, stride=1)
    print(f"sparsity={SPARSITY} width={width} inp={cin} hidden_dim={round(6*cin)} oup={cout} stride={stride}")

    bottleneck = sparseconv.paper.InvertedResidual(
        None, cin, cout, stride, expand_ratio, sparse=True).to(DEVICE)
    bottleneck.debug = True
    x = sampler.gen_input().to(DEVICE)
    mask = sampler.gen_mask().to(DEVICE)
    print(mask.sum() / mask.numel())
    mask_dilate = sampler.gen_expanded_mask(mask).to(DEVICE)
    bottleneck.m_debug = {"std": mask, "dilate": mask_dilate}
    meta = {"gumbel_temp":1.0, "gumbel_noise":True, "save_masks": False}
    v = [x, meta]
    
    # test dense mode
    bottleneck.sparse = False
    with torch.no_grad():
        with utils.Timer(bottleneck) as t:
            for i in range(REPETITION):
                bottleneck.forward(v)
        print(f"Dense  result: {bottleneck.elapsed_time * 1000}ms")

    # test with sparse mode
    bottleneck.sparse = True
    with torch.no_grad():
        with utils.Timer(bottleneck) as t:
            for i in range(REPETITION):
                bottleneck.forward(v)
        print(f"Sparse result: {bottleneck.elapsed_time * 1000}ms")


    
