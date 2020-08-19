
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
from model.Discriminator import Discriminator
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--output_str', type=str, default='test_all', help='dir num of output')
parser.add_argument('--cuda_num', type=int, default=0, help='cuda#')
parser.add_argument('--load_model', type=bool, default=True, help='load model or not')
parser.add_argument('--load_path', type=str, default='./output_test/model_epoch_29.pth', help='path of previous training model')
parser.add_argument('--fea_channel', type=int, default=512, help='channels of features')
parser.add_argument('--mse_hyper', type=int, default=10, help='hyper parameters of mse')
parser.add_argument('--l1_hyper', type=int, default=10, help='hyper parameters of l1')

parser.add_argument('--cp_freq', type=int, default=10, help='how often saving checkpoint')

parser.add_argument('--pre0', type=int, default=0.8, help='hyper parameters vgg layer 0 precept loss')
parser.add_argument('--pre1', type=int, default=0.4, help='hyper parameters vgg layer 1 precept loss')
parser.add_argument('--pre2', type=int, default=0.2, help='hyper parameters vgg layer 2 precept loss')
parser.add_argument('--pre3', type=int, default=0.1, help='hyper parameters vgg layer 3 precept loss')



opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.cuda_num)
    
if not os.path.exists('output_'+opt.output_str):
    os.makedirs('output_'+opt.output_str)

test_folder = 'output_'+opt.output_str


###### Definition of variables ######
# Networks
model = model(fea_channel = opt.fea_channel)
discriminator = Discriminator()

if opt.load_model:
    load_path = opt.load_path
    model.load_state_dict(torch.load(load_path))
    print('model loaded')
    torch.cuda.empty_cache()

model = model.cuda()
discriminator = discriminator.cuda()

vgg_model = models.vgg16(pretrained=True)
vgg_model.cuda()

loss_network = utils.LossNetwork(vgg_model)
loss_network.eval()



# Optimizers & LR schedulers
optimizer_G = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Dataset loader
trainloader = loaddata.getTrainingData(opt.batchSize, size=128)
testloader = loaddata.getTestingData(1,  size=128)


criteria_recon_l2 = nn.MSELoss()
criteria_recon_l1 = nn.L1Loss()

###################################
###### Training ######

with open('./output_'+opt.output_str+'/loss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    
    writer.writerow(['D_real_loss', 'D_fake_loss', 'G_loss', 'recon_loss', 'l2', 'l1', 'preceptual'])
  
    for epoch in range(opt.n_epochs):
        print("epoch = "+str(epoch))
        
        model.train(mode=True)
        discriminator.train(mode=True)

        G_loss_e         = 0
        D_real_loss_e    = 0
        D_fake_loss_e    = 0

        recon_loss_e     = 0
        
        seq_l2_loss_e    = 0
        seq_l1_loss_e    = 0
        seq_preceptual_e = 0

        
        for i, batch in enumerate(trainloader):
            curr_image, curr_depth = batch['image'], batch['depth']
            next_image, next_depth = batch['next_image'], batch['next_depth']

            curr_image = curr_image.cuda()
            curr_depth = curr_depth.cuda()
            next_image = next_image.cuda()
            next_depth = next_depth.cuda()
            
            curr_image_t, curr_depth_t = utils.random_flip(curr_image, curr_depth)
            next_image_t, next_depth_t = utils.random_flip(next_image, next_depth)
            
            # generate random_depth            
            random_index = torch.randperm(opt.batchSize).cuda()
            rand_image   = torch.index_select(curr_image, 0, random_index).detach()
            rand_depth   = torch.index_select(curr_depth, 0, random_index).detach()

            ####################### train discriminator #######################
            # Generate fake image       
            fake_img_curr = model( next_image_t, next_depth_t, curr_depth)
            fake_img_next = model( curr_image_t, curr_depth_t, next_depth)
            rand_img_curr = model( curr_image_t, curr_depth_t, rand_depth)
            fake_img_rand = model(rand_img_curr,   rand_depth, rand_depth)           
            
            # Get scores and loss
            real_d_curr_1, real_d_curr_2 = discriminator(curr_image)
            real_d_next_1, real_d_next_2 = discriminator(next_image)
            fake_d_curr_1, fake_d_curr_2 = discriminator(fake_img_curr)
            fake_d_next_1, fake_d_next_2 = discriminator(fake_img_next)
            rand_d_curr_1, rand_d_curr_2 = discriminator(rand_img_curr)
            fake_d_rand_1, fake_d_rand_2 = discriminator(fake_img_rand)
            
            # mse_loss for LSGAN
            D_loss_fake = utils.lsgan_loss(real_d_curr_1, 1) + utils.lsgan_loss(real_d_next_1, 1) + \
                          utils.lsgan_loss(real_d_curr_2, 1) + utils.lsgan_loss(real_d_next_2, 1)
                          
            D_loss_real = utils.lsgan_loss(fake_d_curr_1, 0) + utils.lsgan_loss(fake_d_next_1, 0) + \
                          utils.lsgan_loss(rand_d_curr_1, 0) + utils.lsgan_loss(fake_d_rand_1, 0) + \
                          utils.lsgan_loss(fake_d_curr_2, 0) + utils.lsgan_loss(fake_d_next_2, 0) + \
                          utils.lsgan_loss(rand_d_curr_2, 0) + utils.lsgan_loss(fake_d_rand_2, 0)

            D_loss = D_loss_fake + D_loss_real 
            
            D_real_loss_e = D_real_loss_e + D_loss_real.item()
            D_fake_loss_e = D_fake_loss_e + D_loss_fake.item()
            
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            
            D_loss.backward(retain_graph=True)
            
            optimizer_D.step()
            
            ####################### train generator #######################
            
            G_loss = utils.lsgan_loss(fake_d_curr_1, 1) + utils.lsgan_loss(fake_d_next_1, 1) + \
                     utils.lsgan_loss(rand_d_curr_1, 1) + utils.lsgan_loss(fake_d_rand_1, 1) + \
                     utils.lsgan_loss(fake_d_curr_2, 1) + utils.lsgan_loss(fake_d_next_2, 1) + \
                     utils.lsgan_loss(rand_d_curr_2, 1) + utils.lsgan_loss(fake_d_rand_2, 1)          
            
            G_loss_e = G_loss_e + G_loss.item()
            
            # reconstruction loss
            l2_loss    = opt.mse_hyper*(criteria_recon_l2(fake_img_curr, curr_image)+criteria_recon_l2(fake_img_next, next_image)+criteria_recon_l2(fake_img_rand, curr_image))
            l1_loss    = opt.l1_hyper *(criteria_recon_l1(fake_img_curr, curr_image)+criteria_recon_l1(fake_img_next, next_image)+criteria_recon_l1(fake_img_rand, curr_image))

            preceptual0_fake = loss_network(fake_img_curr)
            preceptual0_real = loss_network(curr_image)
            preceptual1_fake = loss_network(fake_img_next)
            preceptual1_real = loss_network(next_image)
            preceptual0_rand = loss_network(fake_img_rand)
 
            preceptual  = opt.pre0*criteria_recon_l2(preceptual0_fake[0], preceptual0_real[0])
            preceptual += opt.pre1*criteria_recon_l2(preceptual0_fake[1], preceptual0_real[1])
            preceptual += opt.pre2*criteria_recon_l2(preceptual0_fake[2], preceptual0_real[2])
            preceptual += opt.pre3*criteria_recon_l2(preceptual0_fake[3], preceptual0_real[3])
            
            preceptual += opt.pre0*criteria_recon_l2(preceptual1_fake[0], preceptual1_real[0])
            preceptual += opt.pre1*criteria_recon_l2(preceptual1_fake[1], preceptual1_real[1])
            preceptual += opt.pre2*criteria_recon_l2(preceptual1_fake[2], preceptual1_real[2])
            preceptual += opt.pre3*criteria_recon_l2(preceptual1_fake[3], preceptual1_real[3])
            
            preceptual += opt.pre0*criteria_recon_l2(preceptual0_rand[0], preceptual0_real[0])
            preceptual += opt.pre1*criteria_recon_l2(preceptual0_rand[1], preceptual0_real[1])
            preceptual += opt.pre2*criteria_recon_l2(preceptual0_rand[2], preceptual0_real[2])
            preceptual += opt.pre3*criteria_recon_l2(preceptual0_rand[3], preceptual0_real[3])
            
            img_recon_loss = l2_loss + l1_loss + preceptual
            
            ####################### add up all loss #######################
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            
            total_loss = img_recon_loss + G_loss
            total_loss.backward()
            
            optimizer_G.step()
            
            
            recon_loss_e      = recon_loss_e      + img_recon_loss.item()
            
            seq_l2_loss_e     = seq_l2_loss_e     + l2_loss.item()
            seq_l1_loss_e     = seq_l1_loss_e     + l1_loss.item()
            seq_preceptual_e  = seq_preceptual_e  + preceptual.item()  

            print("===> Epoch[{}]({}/{}): d_fake: {:.4f}, d_real: {:.4f}, g: {:.4f}, image_recon: {:.4f}".format(epoch, i, len(trainloader), D_loss_fake.item(), D_loss_real.item(), G_loss.item(), img_recon_loss.item()))
            # break

        
        if epoch % opt.cp_freq == opt.cp_freq-1:
            torch.save(model.state_dict(), test_folder + '/model_epoch_' + str(epoch) + '.pth') 
            torch.save(discriminator.state_dict(), test_folder + '/D_epoch_' + str(epoch) + '.pth') 
        
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
             

        
        writer.writerow([D_real_loss_e/len(trainloader),  D_fake_loss_e/len(trainloader),  G_loss_e/len(trainloader), 
                          recon_loss_e/len(trainloader),  seq_l2_loss_e/len(trainloader),  seq_l1_loss_e/len(trainloader),  seq_preceptual_e/len(trainloader)])
        csvfile.flush()
        # exit(1)
  
