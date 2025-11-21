import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

from PIL import Image
from IPython.display import display
import cv2
from PIL import ImageFile
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

import glob
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def seed_everything(seed=1234):
    random.seed(seed) # 设置Python内置random模块的种子影响：random.randint(), random.choice(), random.shuffle() 等函数
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置Python哈希种子,作用：确保字典、集合等数据结构的哈希行为一致
    np.random.seed(seed)  # 设置NumPy的随机种子,影响：np.random.rand(), np.random.randint() 等NumPy随机函数
    torch.manual_seed(seed) # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed) # 设置PyTorch的GPU随机种子,影响PyTorch在GPU上的随机操作（如dropout、初始化等）
    torch.backends.cudnn.deterministic = True # 确保CuDNN使用确定性算法,作用：禁用CuDNN的非确定性算法，确保GPU计算结果可重现代价：可能会稍微降低性能
seed_everything(42)

#Read the dataset 
PATH = './dataset/'
labels = pd.read_csv(PATH+'labels.csv')
labelnames = pd.read_csv(PATH + 'sample_submission.csv').keys()[1:] # 获取出第一列以外所有列的名字
print("Train folder has ", len(os.listdir(PATH+'train')),'images which matches with label\'s', len(labels),'images')

from data.dataset import DogBreedDataset,img_transform
batch_size = 12
num_workers = 4
train_img = DogBreedDataset(PATH+'train/', train, transform = img_transform['train'])
valid_img = DogBreedDataset(PATH+'train/', valid, transform = img_transform['valid'])


dataloaders={
    'train':torch.utils.data.DataLoader(train_img, batch_size, num_workers = num_workers, shuffle=True),
    'valid':torch.utils.data.DataLoader(valid_img, batch_size, num_workers = num_workers, shuffle=False)
}

use_cuda = torch.cuda.is_available()

from visualize.plot_loss import plot_multiple_loss_curves_by_epoch
from utils.train import transfer_train
def multi_model_transfer_learning(models,n_epochs):
    train_losses_list=[]
    valid_losses_list=[]
    models_name=[model.__class__.__name__.lower() for model in models]
    for model in models:
        _,train_losses,valid_losses=transfer_train(model,dataloaders,n_epochs=n_epochs)
    train_losses_list.append(train_losses)
    valid_losses_list.append(valid_losses)
    plot_multiple_loss_curves_by_epoch(train_losses_list,valid_losses_list,models_name,n_epochs)


alexnet = models.alexnet(pretrained=True)
googlenet = models.googlenet(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
from models.SeResnet import se_resnext101
senet101=se_resnext101(pretrained="imagenet")

models_list=[alexnet,googlenet,resnet101,vgg16,densenet121,efficientnet_b0,mobilenet_v2,senet101]
multi_model_transfer_learning(models_list,15)

