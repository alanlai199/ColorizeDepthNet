import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from dataloader.nyu_transform import *
import sys
import os

class depthDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.ix[idx, 0]
        depth_name = self.frame.ix[idx, 1]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)

class depthDataset_seq(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.ix[idx, 0]
        depth_name = self.frame.ix[idx, 1]
        
        image = Image.open(image_name)
        depth = Image.open(depth_name)

        file_path = image_name
        file_name = file_path.split('/')
        file_num  = int(file_name[-1].split('.')[0])
        
        next_file_path = ''
        for i in range(len(file_name)-1):
            next_file_path += file_name[i] + '/'
            
        next_img_path = image_name
        next_dep_path = depth_name

        if os.path.isfile(next_file_path + str(file_num+10) + '.jpg'):
            next_img_path = next_file_path + str(file_num+10) + '.jpg'
            next_dep_path = next_file_path + str(file_num+10) + '.png'
        
        elif os.path.isfile(next_file_path + str(file_num-10) + '.jpg'):
            next_img_path = next_file_path + str(file_num-10) + '.jpg'
            next_dep_path = next_file_path + str(file_num-10) + '.png'
           
        next_image = Image.open(next_img_path)
        next_depth = Image.open(next_dep_path)
        

        sample = {'image': image, 'depth': depth, 'next_image': next_image, 'next_depth': next_depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)

def getTrainingData(batch_size=4, size = 128, csv_file='./data/nyu2_train.csv'):
    transformed_training = depthDataset_seq(csv_file=csv_file,
                                            transform=transforms.Compose([
                                                Scale_seq(288),
                                                RandomHorizontalFlip_seq(),
                                                RandomRotate_seq(4),
                                                CenterCrop_seq([256, 256], [256, 256]),
                                                Resize_seq([size, size]),
                                                ToTensor_seq(),
                                            ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=10, pin_memory=False)

    return dataloader_training

def getTestingData(batch_size=4, size = 128, csv_file='./data/nyu2_test.csv'):
    transformed_testing = depthDataset(csv_file=csv_file,
                                       transform=transforms.Compose([
                                           Scale(288),
                                           CenterCrop([256, 256], [256, 256]),
                                           Resize([size, size]),
                                           ToTensor(is_test=True),
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=10, pin_memory=False)

    return dataloader_testing


def getReferanceData(batch_size=4, size = 128, csv_file='./data/nyu2_ref.csv'):
    transformed_testing = depthDataset(csv_file=csv_file,
                                       transform=transforms.Compose([
                                           Scale(288),
                                           CenterCrop([256, 256], [256, 256]),
                                           Resize([size, size]),
                                           ToTensor(is_test=True),
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=10, pin_memory=False)

    return dataloader_testing



