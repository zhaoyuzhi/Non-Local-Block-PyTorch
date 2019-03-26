# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:37:37 2019

@author: ZHAO Yuzhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Spectralnorm import SpectralNorm
from Self_Attn import Self_Attn_FM, Self_Attn_C
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# image_size = 32 or 64
class Generator(nn.Module):
    """DCGAN Generator without Attention"""

    def __init__(self, image_size = 32, z_dim = 100, conv_dim = 64, selfattn = False):
        super(Generator, self).__init__()
        
        self.imsize = image_size
        self.z_dim = z_dim
        self.selfattn = selfattn
        
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        def spectralnorm_block(in_filters, out_filters, normalization = True, firstlayer = False):
            """Returns downsampling layers of each spectralnorm block"""
            layers = []
            if firstlayer:
                layers.append(SpectralNorm(nn.ConvTranspose2d(in_filters, out_filters, 4, 1, 0)))
            else:
                layers.append(SpectralNorm(nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1)))
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.ReLU(inplace = True))
            return layers
        
        # repeat_num = 3 if imsize = 64; repeat_num = 2 if imsize = 32
        repeat_num = int(np.log2(self.imsize)) - 3
        # mult = 8 of repeat_num = 3; mult = 4 of repeat_num = 2
        mult = 2 ** repeat_num
        
        # build layers
        # curr_dim = 512 of image_size = 64; curr_dim = 256 of image_size = 32, and conv_dim = 64
        curr_dim = conv_dim * mult
        self.l1 = nn.Sequential(*spectralnorm_block(z_dim, curr_dim, firstlayer = True))
        self.l2 = nn.Sequential(*spectralnorm_block(curr_dim, int(curr_dim / 2)))
        curr_dim = int(curr_dim / 2)
        self.l3 = nn.Sequential(*spectralnorm_block(curr_dim, int(curr_dim / 2)))
        curr_dim = int(curr_dim / 2)
        # build attention
        if self.selfattn:
            self.attn1 = Self_Attn_FM(curr_dim)
        
        if self.imsize == 64:
            self.l4 = nn.Sequential(*spectralnorm_block(curr_dim, int(curr_dim / 2)))
            curr_dim = int(curr_dim / 2)
            # build attention
            if self.selfattn:
                self.attn2 = Self_Attn_FM(curr_dim)
        
        self.last = nn.Sequential(
                nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1),
                nn.Tanh()
                )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        if self.selfattn:
            if self.imsize == 32:
                out, p1 = self.attn1(out)
                out = self.last(out)
                return out, p1
            if self.imsize == 64:
                out, p1 = self.attn1(out)
                out = self.l4(out)
                out, p2 = self.attn2(out)
                out = self.last(out)
                return out, p1, p2
        else:
            if self.imsize == 64:
                out = self.l4(out)
            out = self.last(out)
            return out

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size = 32, conv_dim = 64, selfattn = False):
        super(Discriminator, self).__init__()
        
        self.imsize = image_size
        self.selfattn = selfattn
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        def spectralnorm_block(in_filters, out_filters, normalization = False):
            """Returns downsampling layers of each spectralnorm block"""
            layers = []
            layers.append(SpectralNorm(nn.Conv2d(in_filters, out_filters, 4, 2, 1)))
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers
        
        self.l1 = nn.Sequential(*spectralnorm_block(3, conv_dim, normalization = False))
        self.l2 = nn.Sequential(*spectralnorm_block(conv_dim, (conv_dim * 2)))
        self.l3 = nn.Sequential(*spectralnorm_block((conv_dim * 2), (conv_dim * 4)))
        # build attention
        if self.selfattn:
            self.attn1 = Self_Attn_FM(conv_dim * 4)
        self.last = nn.Conv2d(conv_dim * 4, 1, 4)
        
        if self.imsize == 64:
            self.l4 = nn.Sequential(*spectralnorm_block((conv_dim * 4), (conv_dim * 8)))
            # build attention
            if self.selfattn:
                self.attn2 = Self_Attn_FM(conv_dim * 8)
            self.last = nn.Conv2d(conv_dim * 8, 1, 4)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        if self.selfattn:
            if self.imsize == 32:
                out, p1 = self.attn1(out)
                out = self.last(out)
                return out.squeeze(), p1
            if self.imsize == 64:
                out, p1 = self.attn1(out)
                out = self.l4(out)
                out, p2 = self.attn2(out)
                out = self.last(out)
                return out.squeeze(), p1, p2
        else:
            if self.imsize == 64:
                out = self.l4(out)
            out = self.last(out)
            return out.squeeze()
