import torch
import torch.nn as nn
import torch.nn.functional as F

import dynconv



BN_MOMENTUM = 0.1

class InvertedResidual(nn.Module):
    expansion = 2
    def __init__(self, cfg, inp, oup, stride=1, expand_ratio=6, sparse=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        self.sparse = sparse
        # print(f'Inverted Residual - sparse: {sparse}: inp {inp}, hidden_dim {hidden_dim}, ' + 
        #       f'oup {oup}, stride {stride}, expand_ratio {expand_ratio}')
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=BN_MOMENTUM),
        ])
        self.conv = nn.Sequential(*layers)
        
        if sparse:
            assert self.identity
            assert expand_ratio != 1
            self.masker = dynconv.MaskUnit(inp, stride=stride, dilate_stride=stride)
        else:
            self.masker = None

    def forward(self, v):
        x, meta = v
        if not self.sparse:
            out = self.conv(x)
            if self.identity:
                out += x
            return out, meta
        else:    
            assert self.identity and self.expand_ratio != 1
            m, meta = self.masker(x, meta)
            # use prepared mask for more precise timing
            if hasattr(self, "debug_m") and self.debug is True:
                m = self.debug_m
            mask, mask_dilate = m['std'], m['dilate']

            fast_inference = not self.training

            out = x.clone() # clone should not be needed, but otherwise seems to be bugged
            if fast_inference:
                x = dynconv.gather(x, mask_dilate)
            
            x = dynconv.conv1x1(   self.conv[0],               x, mask_dilate,       fast=fast_inference)
            x = dynconv.bn_relu(   self.conv[1], self.conv[2], x, mask_dilate,       fast=fast_inference)
            x = dynconv.conv3x3_dw(self.conv[3],               x, mask_dilate, mask, fast=fast_inference)
            x = dynconv.bn_relu(   self.conv[4], self.conv[5], x, mask,              fast=fast_inference)
            x = dynconv.conv1x1(   self.conv[6],               x, mask,              fast=fast_inference)
            x = dynconv.bn_relu(   self.conv[7], None,         x, mask,              fast=fast_inference)
            
            if fast_inference:
                out = dynconv.scatter(x, out, mask, sum_out=True)
            else:
                out = out + dynconv.apply_mask(x, mask)
            return out, meta