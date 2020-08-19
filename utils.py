import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
import torchvision.models as models
import random

from torch import nn

criteria_recon_l1 = nn.L1Loss()


'''----------random_flip----------'''
def random_flip(image, depth):
    temp_i, temp_d = image.detach(), depth.detach()
    
    if random.random() < 0.5:
        temp_i = torch.flip(temp_i, [2])
        temp_d = torch.flip(temp_d, [2])
        
    if random.random() < 0.5:
        temp_i = torch.flip(temp_i, [3])
        temp_d = torch.flip(temp_d, [3])
    
    return temp_i, temp_d


'''
if __name__ == "__main__":
    image = torch.randn(1, 3, 2, 2)
    depth = torch.randn(1, 1, 2, 2)
    image, depth = image.cuda(), depth.cuda()
    
    out_i, out_d = random_flip(image, depth)    
    
    print(image[0,0])
    print(out_i[0,0])
    
    print(depth[0,0])
    print(out_d[0,0])
'''

'''----------vgg_preceptual----------'''
LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)


'''
if __name__ == "__main__":
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.cuda()

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    image = torch.randn(16, 3, 128, 128)
    depth = torch.randn(16, 1, 128, 128)
    image, depth = image.cuda(), depth.cuda()


    out = loss_network(image)
    # print(out[3].shape)
    print(len(out))


'''

'''----------clamp_depth----------'''
def clamp(depth_map, max_depth):
    max_depth *= 100
    mask = depth_map < max_depth
    clamp_depth = torch.clamp(depth_map, 0, max_depth)/max_depth
    norm_depth = (clamp_depth-0.5)/0.5
    return norm_depth, mask
    

'''----------show_result----------'''
cmap = plt.cm.jet

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

def output_rgb(rgb):
    rgb_cpu = 255 * np.transpose(np.squeeze(rgb.cpu().detach().numpy()), (1, 2, 0))  # H, W, C
    return rgb_cpu
    
def output_depth(depth, state = 'colored'):
    depth_cpu = np.squeeze(depth.cpu().detach().numpy())
    if state == 'colored':
        depth_cpu = colored_depthmap(depth_cpu, np.min(depth_cpu), np.max(depth_cpu))
    elif state == 'clamped':
        depth_cpu = (depth_cpu*0.5+0.5)
        depth_cpu = np.expand_dims(depth_cpu, axis=2)
        depth_cpu = np.repeat(depth_cpu, 3, axis=2)*255
    else:
        depth_cpu = depth_cpu / 2**16
        depth_cpu = np.expand_dims(depth_cpu, axis=2)
        depth_cpu = np.repeat(depth_cpu, 3, axis=2)*255
    
    return depth_cpu



def merge_into_row(input):
    img_merge = np.hstack(input)
    return img_merge

def merge_into_image(image, row):
    return np.vstack([image, row])

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
    



'''------------lsgan_loss--------------------'''
def lsgan_loss(score, target=1):
    dtype = type(score)
    
    if target == 1:
        label = torch.ones(score.size()).cuda()
    elif target == 0:
        label = torch.zeros(score.size()).cuda()
    
    criterion = nn.MSELoss()
    loss = criterion(score, label)
    
    return loss


'''-----------surface normals------------------'''
def surface_normal(d_im):   
    zy, zx = np.gradient(d_im*10, axis=[2, 3])  

    normal = np.concatenate((-zx, -zy, np.ones_like(d_im)), axis=1)
    n = np.linalg.norm(normal, axis=1)

    normal[:, 0, :, :] /= n
    normal[:, 1, :, :] /= n
    normal[:, 2, :, :] /= n
    
    return normal


if __name__ == "__main__":

    
    depth = np.random.randn(16, 1, 256, 256)

    print(surface_normal(depth).shape)
    
    



