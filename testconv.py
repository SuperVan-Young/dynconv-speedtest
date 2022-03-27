from curses.panel import bottom_panel
from logging import warning
import os
from tkinter.tix import Tree

import sparseconv.paper
import sparseconv.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.profiler as profiler

SPARSITY = 0.5
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

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

# with torch.profiler.profile(
#     activities=[
#         # torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],

#     # In this example with wait=1, warmup=1, active=2,
#     # profiler will skip the first step/iteration,
#     # start warming up on the second, record
#     # the third and the forth iterations,
#     # after which the trace will become available
#     # and on_trace_ready (when set) is called;
#     # the cycle repeats starting with the next step

#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=2),
#     on_trace_ready=trace_handler
#     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
#     # used when outputting for tensorboard
#     ) as p:

for testcase in testcases:
    width, cin, expand_ratio, cout, stride = testcase
    sampler = utils.ConvSampler(
        channel_in=cin, channel_out=cout, sparsity=SPARSITY, stride=1)
    print(f"sparsity={SPARSITY} width={width} inp={cin} hidden_dim={round(6*cin)} oup={cout} stride={stride}")

    bottleneck = sparseconv.paper.InvertedResidual(
        None, cin, cout, stride, expand_ratio, sparse=True).cuda()
    bottleneck.debug = True
    bottleneck.eval()
    x = sampler.gen_input()
    m = sampler.gen_mask()
    bottleneck.m_debug = m
    meta = {"gumbel_temp":1.0, "gumbel_noise":True, "save_masks": False}
    v = [x, meta]

    print(m["std"])
    print(m['dilate'])
    
    # test dense mode
    bottleneck.sparse = False
    with torch.no_grad():
        for i in range(REPETITION):
            bottleneck.forward(v)
        with utils.Timer(bottleneck) as t:
            for i in range(REPETITION):
                bottleneck.forward(v)
        print(f"Dense  result: {bottleneck.elapsed_time * 1000}ms")
        # p.step()
        

    # test with sparse mode
    bottleneck.sparse = True
    with torch.no_grad():
        for i in range(REPETITION):
            bottleneck.forward(v)
        with utils.Timer(bottleneck) as t:
            for i in range(REPETITION):
                bottleneck.forward(v)
        print(f"Sparse result: {bottleneck.elapsed_time * 1000}ms")
        # p.step()
    print()



