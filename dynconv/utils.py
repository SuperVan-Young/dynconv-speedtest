from distutils.log import warn
import time
import torch

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
if device == "cpu":
    warn("Torch is using CPU, the measurement may be incorrect.")


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

    def __init__(self, batch_size=16, channel_in=3, channel_out=96,
                 kh=3, kw=3, ih=224, iw=224, sparsity=0.5) -> None:
        self.batch_size = batch_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kh = kh
        self.kw = kw
        self.ih = ih
        self.iw = iw
        self.sparsity = sparsity

    def gen_weight(self):
        return torch.randn(size=(self.channel_out, self.channel_in, self.kh, self.kw)).to(device)
    
    def gen_input(self):
        return torch.randn(size=(self.batch_size, self.channel_in, self.ih, self.iw)).to(device)

    def gen_mask(self):
        mask = torch.randint(0, 100, size=(self.batch_size, self.channel_in, self.ih, self.iw)) / 100.0
        return (mask < self.sparsity).type(torch.uint8).to(device)
    
    def gen_bias(self):
        return torch.randn(self.channel_out).to(device)
