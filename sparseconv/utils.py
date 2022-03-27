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
        self.expandmask = dynconv.maskunit.ExpandMask(stride).cuda()

    def gen_weight(self):
        return torch.randn(size=(self.channel_out, self.channel_in, self.kh, self.kw)).cuda()

    def gen_input(self):
        return torch.randn(size=(self.batch_size, self.channel_in, self.ih, self.iw)).cuda()

    def gen_mask(self):
        hard = torch.randint(0, 1000, size=(
            self.batch_size, 1, self.ih, self.iw)) / 1000.0
        hard = (hard > self.sparsity).type(torch.uint8).cuda()
        mask = dynconv.Mask(hard, None)

        hard_dilate = self.expandmask(mask.hard).cuda()
        mask_dilate = dynconv.Mask(hard_dilate, None)

        mask.active_positions_list = dynconv.make_active_positions_list(mask.hard)
        mask_dilate.active_positions_list = dynconv.make_active_positions_list(mask_dilate.hard)
        mask_dilate.active_positions_list_inverted = dynconv.make_active_positions_list_inverted(mask_dilate.hard, mask_dilate.active_positions_list)
        
        m = {'std': mask, 'dilate': mask_dilate}
        return m        

    def gen_bias(self):
        return torch.randn(self.channel_out).cuda()
