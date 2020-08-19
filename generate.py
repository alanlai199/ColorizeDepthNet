
import argparse
import itertools
import os
from time import localtime, strftime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision.models as models

import matplotlib.image
import dataloader.loaddata as loaddata

import numpy as np
import sys
import csv
import math

from model.ColorizeDepthNet import model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--output_str', type=str, default='generated_data', help='dir num of output')
parser.add_argument('--cuda_num', type=int, default=1, help='cuda#')
parser.add_argument('--load_model', type=bool, default=True, help='load model or not')
parser.add_argument('--load_path', type=str, default='./output_test/model_epoch_59.pth', help='the output directory of previous training model')
parser.add_argument('--fea_channel', type=int, default=512, help='channels of features')
parser.add_argument('--num_of_ref', type=int, default=5, help='number of reference pairs')



opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.cuda_num)

if not os.path.exists(opt.output_str):
    os.makedirs(opt.output_str)

test_folder = opt.output_str
# Networks
G = model(fea_channel = opt.fea_channel)

load_path = opt.load_path
G.load_state_dict(torch.load(load_path))
print('model loaded')
torch.cuda.empty_cache()

G = G.cuda()
G.eval()

testloader = loaddata.getTestingData(opt.num_of_ref)
refloader = loaddata.getReferanceData(opt.num_of_ref)

for i, batch in enumerate(refloader):
    rand_image, rand_depth = batch['image'].cuda() , batch['depth'].cuda()
    break

for i, batch in enumerate(testloader):
    image, depth = batch['image'], batch['depth']
    
    real_image = image.cuda() 
    real_depth = depth.cuda()
    
    for i_s in range(real_image.shape[0]):
        # reconstruction
        rgb = G(real_image[i_s:i_s+1], real_depth[i_s:i_s+1], real_depth[i_s:i_s+1])  
        rgb = 255*np.transpose(np.squeeze(rgb.cpu().detach().numpy()),(1,2,0))
        filename = test_folder + '/' + str(i*opt.num_of_ref + i_s) + '_recon.png'
        utils.save_image(rgb, filename) 
        
        # with reference pair
        for i_t in range(opt.num_of_ref):
            rgb = G(rand_image[i_t:i_t+1], rand_depth[i_t:i_t+1], real_depth[i_s:i_s+1])  
            rgb = 255*np.transpose(np.squeeze(rgb.cpu().detach().numpy()),(1,2,0))
            filename = test_folder + '/' + str(i*opt.num_of_ref + i_s) + '_' + str(i_t + 1) + '.png'
            utils.save_image(rgb, filename) 


'''
testloader = loaddata.getTestingData(1)
refloader = loaddata.getReferanceData(1)

for j, batch_t in enumerate(testloader):
    real_image, real_depth = batch_t['image'].cuda(), batch_t['depth'].cuda()
    
    rgb = G(real_image, real_depth, real_depth)  
    rgb = 255*np.transpose(np.squeeze(rgb.cpu().detach().numpy()),(1,2,0))
    filename = test_folder + '/' + str(j) + '_0.png'
    utils.save_image(rgb, filename) 
    
    for i, batch_r in enumerate(refloader):
        ref_image, ref_depth = batch_r['image'].cuda(), batch_r['depth'].cuda()

        rgb = G(ref_image, ref_depth, real_depth)  
        rgb = 255*np.transpose(np.squeeze(rgb.cpu().detach().numpy()),(1,2,0))
        filename = test_folder + '/' + str(j) + '_' + str(i+1) + '.png'
        utils.save_image(rgb, filename) 
'''
