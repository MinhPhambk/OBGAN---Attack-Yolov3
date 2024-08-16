import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torch.nn import init
import functools
from activations import *

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            # 3x299x299
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),           
            # c8 8x148x148
            nn.Conv2d(8, 16, kernel_size=8, stride=4, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),           
            # c16 16x36x36 
            nn.Conv2d(16, 32, kernel_size=8, stride=4, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),               
            # c32 32 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),  
            # c64 64 x 3 x 3
            nn.Conv2d(64, 1, 2, bias=True),
            #c1 1x2x2 
        ]
        self.model = nn.Sequential(*model)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x).squeeze() 
        probs = self.prob(output)
        return output, probs
class Generator(nn.Module):
    '''
    For 299*299 images.
    '''
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # c7s1-8 3*299*299
            nn.Conv2d(gen_input_nc, 8, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # d16 8*293*293
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # d32 16*145*145
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # d64 32*71*71
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #d64 64*34*34
        ]

        #r64*4  64x34x34
        bottle_neck_lis = [ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),]
        
        decoder_lis = [
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # u16 32*71*71
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # u8 16*145*145
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # c7s1-3 8*293*293
            nn.ConvTranspose2d(8, image_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
            # 3*299*299
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)     

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)
                       ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out