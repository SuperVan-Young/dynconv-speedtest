from distutils.log import warn
import time
import torch
import dynconv

import os


class Timer():

    def __init__(self, run) -> None:
        self.start = None
        self.end = None
        self.run = run
        self.run.elapsed_time = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.run.elapsed_time = self.end - self.start


class ConvSampler():

    def __init__(self, batch_size=1, channel_in=3, channel_out=96,
                 kh=3, kw=3, ih=224, iw=224, stride=1, sparsity=0.5) -> None:
        self.batch_size = batch_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kh = kh
        self.kw = kw
        self.ih = ih
        self.iw = iw
        self.stride = stride
        self.sparsity = sparsity
        self.expandmask = dynconv.maskunit.ExpandMask(stride)

    def gen_weight(self):
        return torch.randn(size=(self.channel_out, self.channel_in, self.kh, self.kw))

    def gen_input(self):
        return torch.randn(size=(self.batch_size, self.channel_in, self.ih, self.iw))

    def gen_mask(self):
        mask = torch.randint(0, 1000, size=(
            self.batch_size, 1, self.ih, self.iw)) / 1000.0
        return (mask > self.sparsity).type(torch.uint8)

    def gen_expanded_mask(self, mask):
        dilated_mask = self.expandmask(mask)
        return dilated_mask

    def gen_bias(self):
        return torch.randn(self.channel_out)
