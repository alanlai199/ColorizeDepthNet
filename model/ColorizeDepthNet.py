import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torchvision
import numpy as np

import os
from os.path import join

import sys

import torchvision.models as models

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class encoder_t(nn.Module):
    def __init__(self, block=ResidualBlock, in_channels=4, l_channels=32, layers=[4,4,4,4]):
        super(encoder_t, self).__init__()
        self.in_channels = in_channels
        self.l_channels = l_channels
        
        self.conv = nn.Conv2d(in_channels, l_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.bn = nn.BatchNorm2d(l_channels)
        self.relu = nn.ReLU(inplace=True) #out:  16, 64, 64       
        
        self.layer1   = self.make_layer(block, l_channels*  2, layers[0])      #out:  32, 64, 64
        self.layer2   = self.make_layer(block, l_channels*  4, layers[1], 2)   #out:  64, 32, 32
        self.layer3   = self.make_layer(block, l_channels*  8, layers[2], 2)   #out: 128, 16, 16
        self.layer4   = self.make_layer(block, l_channels* 16, layers[3], 2)   #out: 256,  8,  8

        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.l_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.l_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.l_channels, out_channels, stride, downsample))
        
        self.l_channels = out_channels
        
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):

        a = self.conv(x)    
        a = self.bn(a)
        a = self.relu(a)
        a = self.layer1(a)    
        a = self.layer2(a)    
        a = self.layer3(a)
        a = self.layer4(a)
        
        return a


class encoder_d(nn.Module):
    def __init__(self, block=ResidualBlock, in_channels=1, l_channels=32, layers=[4,4,4,4]):
        super(encoder_d, self).__init__()
        self.in_channels = in_channels
        self.l_channels = l_channels
        
        self.conv = nn.Conv2d(in_channels, l_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.bn = nn.BatchNorm2d(l_channels)
        self.relu = nn.ReLU(inplace=True) #out:  16, 64, 64       
        
        self.layer1 = self.make_layer(block, l_channels*  2, layers[0])      #out:  32, 64, 64
        self.layer2 = self.make_layer(block, l_channels*  4, layers[1], 2)   #out:  64, 32, 32
        self.layer3 = self.make_layer(block, l_channels*  8, layers[2], 2)   #out: 128, 16, 16
        self.layer4 = self.make_layer(block, l_channels* 16, layers[3], 2)   #out: 256,  8,  8
        
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.l_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.l_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.l_channels, out_channels, stride, downsample))
        self.l_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):

        a    = self.conv(x)
        a    = self.bn(a)
        out1 = self.relu(a)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out  = self.layer4(out4)
        
        return out, out1, out2, out3, out4


      
class decoder(nn.Module):
    def __init__(self, block=ResidualBlock, d_fSize=512, layers=[4,4,4,4]):
        super(decoder, self).__init__()
        
        fSize = d_fSize
        
        # input size 256*8*8
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(d_fSize*2, fSize//2, 4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(fSize//2),
            nn.ReLU()
        )
        
        self.l_channels = fSize//2
        self.layer1 = self.make_layer(block, fSize//2, layers[0])      #out:  128, 16, 16
        
        # state size 128*16*16
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(fSize, fSize//4, 4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(fSize//4),
            nn.ReLU()
        )
        
        self.l_channels = fSize//4
        self.layer2 = self.make_layer(block, fSize//4, layers[1])      #out:   64, 32, 32
        
        # state size 64*32*32
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(fSize//2, fSize//8, 4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(fSize//8),
            nn.ReLU()
        )
        
        self.l_channels = fSize//8
        self.layer3 = self.make_layer(block, fSize//8, layers[2])      #out:   32, 64, 64
        
        # state size 32*64*64
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(fSize//4, fSize//16, 4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(fSize//16),
            nn.ReLU()
        )
        
        self.l_channels = fSize//16
        self.layer4 = self.make_layer(block, fSize//16, layers[3])      #out:   16, 128, 128
        
        # state size 16*128*128
        
        

        self.output = nn.Sequential(
            nn.Conv2d(fSize//16, 3, 1),
            nn.Sigmoid()
        )


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.l_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.l_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.l_channels, out_channels, stride, downsample))
        self.l_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


        
    def forward(self, x, x1, x2, x3, x4):

        y = self.conv1(x)
        y = self.layer1(y)
        y = torch.cat((y, x4), 1)
        y = self.conv2(y)
        y = self.layer2(y)
        y = torch.cat((y, x3), 1)
        y = self.conv3(y)
        y = self.layer3(y)
        y = torch.cat((y, x2), 1)
        y = self.conv4(y)
        y = self.layer4(y)
           
        out = self.output(y)
        
        return out
        
        
class model(nn.Module):
    def __init__(self, fea_channel = 512):
        super(model, self).__init__()  
        
        self.encoder_t = encoder_t(in_channels=4, l_channels=fea_channel//16)
        self.encoder_d = encoder_d(in_channels=1, l_channels=fea_channel//16)
        
        self.decoder = decoder(d_fSize = fea_channel)

    
    def forward(self, RGB_t, depth_t, depth_d):
        # print(RGB_t.shape)
        # print(depth_t.shape)
        # print(depth_d.shape)
    
        in_t = torch.cat((RGB_t, depth_t), 1)
        
        texture = self.encoder_t(in_t)
        out, out1, out2, out3, out4 = self.encoder_d(depth_d)
  
        out = torch.cat([out, texture], dim=1)
        
        out = self.decoder(out, out1, out2, out3, out4)

        return out
        
    def give_encoder(self):
        return self.encoder_t

        
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp        
        
        
if __name__ == "__main__":
    model = model()
    model = model.cuda()
    
    image = torch.randn(4, 3, 128, 128)
    depth = torch.randn(4, 1, 128, 128)
    image, depth = image.cuda(), depth.cuda()
    
    model.train()
    
    with torch.no_grad():
        out = model(image,depth,depth)
        print(out.shape)
        
        
