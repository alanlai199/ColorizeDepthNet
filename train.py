
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

from model.ColorizeDepthNet import model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--output_str', type=str, default='test', help='dir num of output')
parser.add_argument('--cuda_num', type=int, default=3, help='cuda#')
parser.add_argument('--load_model', type=bool, default=False, help='load model or not')
parser.add_argument('--load_path', type=str, default=None, help='path of previous training model')
parser.add_argument('--fea_channel', type=int, default=512, help='channels of features')
parser.add_argument('--mse_hyper', type=int, default=100, help='hyper parameters of mse')
parser.add_argument('--l1_hyper', type=int, default=100, help='hyper parameters of l1')

parser.add_argument('--cp_freq', type=int, default=10, help='how often saving checkpoint')

parser.add_argument('--pre0', type=int, default=8, help='hyper parameters vgg layer 0 precept loss')
parser.add_argument('--pre1', type=int, default=4, help='hyper parameters vgg layer 1 precept loss')
parser.add_argument('--pre2', type=int, default=2, help='hyper parameters vgg layer 2 precept loss')
parser.add_argument('--pre3', type=int, default=1, help='hyper parameters vgg layer 3 precept loss')



opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.cuda_num)
    
if not os.path.exists('output_'+opt.output_str):
    os.makedirs('output_'+opt.output_str)

test_folder = 'output_'+opt.output_str


###### Definition of variables ######
# Networks
model = model(fea_channel = opt.fea_channel)

if opt.load_model:
    load_path = opt.load_path
    model.load_state_dict(torch.load(load_path))
    print('model loaded')

model = model.cuda()

vgg_model = models.vgg16(pretrained=True)
vgg_model.cuda()

loss_network = utils.LossNetwork(vgg_model)
loss_network.eval()



# Optimizers & LR schedulers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Dataset loader
trainloader = loaddata.getTrainingData(opt.batchSize, size=128)
testloader = loaddata.getTestingData(1,  size=128)




criteria_recon_l2 = nn.MSELoss()
criteria_recon_l1 = nn.L1Loss()

###################################
###### Training ######

with open('./output_'+opt.output_str+'/loss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    
    writer.writerow(['total_loss', 'l2', 'l1', 'preceptual'])
  
    for epoch in range(opt.n_epochs):
        print("epoch = "+str(epoch))
        
        model.train(mode=True)

        total_loss_e = 0
        
        seq_l2_loss_e = 0
        seq_l1_loss_e = 0
        seq_preceptual_e = 0

        
        for i, batch in enumerate(trainloader):
            image, depth = batch['image'], batch['depth']
            next_image, next_depth = batch['next_image'], batch['next_depth']

            real_image = image.cuda()
            real_depth = depth.cuda()
            
            next_image = next_image.cuda()
            next_depth = next_depth.cuda()
            
            real_image_t, real_depth_t = utils.random_flip(real_image, real_depth)
            next_image_t, next_depth_t = utils.random_flip(next_image, next_depth)
            
            # generate random_depth            
            # random_index   = torch.randperm(opt.batchSize).cuda()
            # random_image   = torch.index_select(real_image, 0, random_index).detach()
            # random_depth   = torch.index_select(real_depth, 0, random_index).detach()


            optimizer.zero_grad()
            
            ####################### train seq data #######################
            rgb0 = model(real_image_t, real_depth_t, next_depth)
            rgb1 = model(next_image_t, next_depth_t, real_depth)

            preceptual0_fake = loss_network(rgb0)
            preceptual0_real = loss_network(next_image)
            
            preceptual1_fake = loss_network(rgb1)
            preceptual1_real = loss_network(real_image)
            
            # reconstruction loss
            seq_l2_loss     = opt.mse_hyper*criteria_recon_l2(rgb0, next_image)
            seq_l2_loss    += opt.mse_hyper*criteria_recon_l2(rgb1, real_image)
            
            seq_l1_loss     = opt.l1_hyper*criteria_recon_l1(rgb0, next_image)
            seq_l1_loss    += opt.l1_hyper*criteria_recon_l1(rgb1, real_image)
            
            seq_preceptual  = opt.pre0*criteria_recon_l2(preceptual0_fake[0], preceptual0_real[0])
            seq_preceptual += opt.pre1*criteria_recon_l2(preceptual0_fake[1], preceptual0_real[1])
            seq_preceptual += opt.pre2*criteria_recon_l2(preceptual0_fake[2], preceptual0_real[2])
            seq_preceptual += opt.pre3*criteria_recon_l2(preceptual0_fake[3], preceptual0_real[3])
            
            seq_preceptual  = opt.pre0*criteria_recon_l2(preceptual1_fake[0], preceptual1_real[0])
            seq_preceptual += opt.pre1*criteria_recon_l2(preceptual1_fake[1], preceptual1_real[1])
            seq_preceptual += opt.pre2*criteria_recon_l2(preceptual1_fake[2], preceptual1_real[2])
            seq_preceptual += opt.pre3*criteria_recon_l2(preceptual1_fake[3], preceptual1_real[3])
            
            seq_loss = seq_l2_loss + seq_l1_loss + seq_preceptual
            
            ####################### add up all loss #######################
            total_loss = seq_loss
            total_loss.backward()
            
            optimizer.step()
            
            
            total_loss_e      = total_loss_e      + total_loss.item()
            
            seq_l2_loss_e     = seq_l2_loss_e     + seq_l2_loss.item()
            seq_l1_loss_e     = seq_l1_loss_e     + seq_l1_loss.item()
            seq_preceptual_e  = seq_preceptual_e  + seq_preceptual.item()  

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i, len(trainloader), total_loss.item()))
            # break
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, total_loss_e / len(trainloader)))

        
        if epoch % opt.cp_freq == opt.cp_freq-1:
            torch.save(model.state_dict(), test_folder + '/model_epoch_' + str(epoch) + '.pth') 
        
        ####testing
        model.eval()

        skip = len(testloader) // 9  # save images every skip iters
        
        with torch.no_grad():
            for i, batch in enumerate(testloader):

                if i == 0:
                    image, depth = batch['image'], batch['depth']
                
                    real_image, real_depth = image.cuda(), depth.cuda()
                    rand_image, rand_depth = image.cuda(), depth.cuda() 

                    rgb0 = model(real_image, real_depth, real_depth)
                    rgb1 = model(real_image, real_depth, rand_depth)
                    rgb2 = model(rand_image, rand_depth, real_depth)
                  
                    rgb_ori = utils.output_rgb(real_image)   
                    dep_ori = utils.output_depth(real_depth)
                    out_ori = utils.output_rgb(rgb0)
                    dep_ran = utils.output_depth(rand_depth)
                    out_dep = utils.output_rgb(rgb1)
                    rgb_ran = utils.output_rgb(rand_image)
                    out_rgb = utils.output_rgb(rgb2)
                  
                    img_merge = utils.merge_into_row([rgb_ori, dep_ori, out_ori, dep_ran, out_dep, rgb_ran, out_rgb])
                
                elif (i < 8 * skip) and (i % skip == 0):
                    image, depth = batch['image'], batch['depth']
                    real_image, real_depth = image.cuda(), depth.cuda()
                    
                    rgb0 = model(real_image, real_depth, real_depth)
                    rgb1 = model(real_image, real_depth, rand_depth)
                    rgb2 = model(rand_image, rand_depth, real_depth)
                  
                    rgb_ori = utils.output_rgb(real_image)   
                    dep_ori = utils.output_depth(real_depth)
                    out_ori = utils.output_rgb(rgb0)
                    dep_ran = utils.output_depth(rand_depth)
                    out_dep = utils.output_rgb(rgb1)
                    rgb_ran = utils.output_rgb(rand_image)
                    out_rgb = utils.output_rgb(rgb2)
                  
                    row = utils.merge_into_row([rgb_ori, dep_ori, out_ori, dep_ran, out_dep, rgb_ran, out_rgb])
                    img_merge = utils.merge_into_image(img_merge, row)
                    
                elif i == 8 * skip:                    
                    filename = test_folder + '/comparison_' + str(epoch) + '.png'
                    utils.save_image(img_merge, filename) 
             

        
        writer.writerow([ total_loss_e/len(trainloader), 
                         seq_l2_loss_e/len(trainloader),  seq_l1_loss_e/len(trainloader),  seq_preceptual_e/len(trainloader)])
        csvfile.flush()
        # exit(1)
  
