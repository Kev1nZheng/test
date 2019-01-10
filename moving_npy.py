import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import shutil

root_dir = '/home/shangqian/workspace/LSTM_Memory/'
npy_dir = os.path.join(root_dir,'Data/OF_feature/')

train_list = os.path.join(root_dir,'UCF_list/trainlist01.txt')
val_list = os.path.join(root_dir,'UCF_list/testlist01.txt')

train_dir = os.path.join(root_dir,'Data/Flow/train/')
val_dir = os.path.join(root_dir,'Data/Flow/val/')

action_label = {}
num_class = 0
with open(train_list) as f:
    content = f.readlines()
    content = [x.strip('\r\n') for x in content]
    f.close()
num_class = 0
print('Moving train set...')
for label in content:
    video_name = label.split('/')[-1]
    floder_name = video_name.split('.avi')[0]
    video_class = floder_name.split('_')[1]

    old_npy_dir = npy_dir + video_class + '/' + floder_name + '.npy'
    new_npy_dir = train_dir + video_class + '/' + floder_name + '.npy'
    old_dir = npy_dir + video_class
    new_dir = train_dir + video_class
    oldpathExists = os.path.exists(old_dir)
    # print(oldpathExists)
    if oldpathExists:
        newpathExists = os.path.exists(new_dir)
        if not newpathExists:
            os.mkdir(new_dir)
            num_class = num_class + 1
            print(video_class, num_class)
        for file in os.listdir(old_dir):
            fileExists = os.path.exists(new_npy_dir)
            if not fileExists:
                f = open(new_npy_dir, 'a')
                shutil.copyfile(old_npy_dir, new_npy_dir)
            else:
                break
with open(val_list) as f:
    content = f.readlines()
    content = [x.strip('\r\n') for x in content]
    f.close()
num_class = 0
print()
print('Moving val set...')
for label in content:
    video_name = label.split('/')[-1]
    floder_name = video_name.split('.avi')[0]
    video_class = floder_name.split('_')[1]
    old_npy_dir = npy_dir + video_class + '/' + floder_name + '.npy'
    new_npy_dir = val_dir + video_class + '/' + floder_name + '.npy'
    old_dir = npy_dir + video_class
    new_dir = val_dir + video_class
    oldpathExists = os.path.exists(old_dir)
    # print(oldpathExists)
    if oldpathExists:
        newpathExists = os.path.exists(new_dir)
        if not newpathExists:
            os.mkdir(new_dir)
            num_class = num_class + 1
            print(video_class, num_class)
        for file in os.listdir(old_dir):
            fileExists = os.path.exists(new_npy_dir)
            if not fileExists:
                f = open(new_npy_dir, 'a')
                shutil.copyfile(old_npy_dir, new_npy_dir)
            else:
                break