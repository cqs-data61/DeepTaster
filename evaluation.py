"""
Evaluation
"""
import argparse
import numpy as np
import easydict 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchvision.datasets as dset
from deepsvdd import DeepSVDD_network, pretrain_autoencoder, TrainerDeepSVDD
import os

parser = argparse.ArgumentParser(description='DFT image generation')
parser.add_argument('--test', default='./DFTimages/test', type=str, help='test dataset path')
# parser.add_argument('--test', default='Resnet101', type=str, help='test dataset path')
# parser.add_argument('--val', default='cifar100', type=str, help='validation dataset path')
parser.add_argument('--classifier_dir', default='./Deepsvdd', type=str, help='classifier directory')

if __name__ == '__main__':
  opt = parser.parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = dset.ImageFolder(root=opt.test,
                             transform=transforms.Compose([
                                 transforms.Grayscale(),
                                 transforms.Resize(28),      
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor(),    
                             ]))

  test_dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=8)

  net = DeepSVDD_network().to(device)
  state_dict = torch.load(opt.classifier_dir+'/pretrained.pth')
  c = torch.Tensor(state_dict['center']).to(device)
  net=torch.load(opt.classifier_dir+'/deepsvdd.pt')
  threshold_file=open(opt.classifier_dir+'/threshold.txt','r')
  threshold=float(threshold_file.readline())
  threshold_file.close()


  scores = []
  net.eval()
  print('Testing...')
  pos_image_max=0
  neg_image_max=0
  pos_total=0
  neg_total=0
  with torch.no_grad():
      for x, y in test_dataloader:
          x = x.float().to(device)
          z = net(x)
          score = torch.sum((z - c) ** 2, dim=1)
          for i in range(16):
            if y[i]==0:
              pos_total+=1
              if score[i]<=threshold:
                pos_image_max+=1
            elif y[i]==1:
              neg_total+=1
              if score[i]>threshold:
                neg_image_max+=1
            if neg_total%288==0:
              print(neg_image_max,neg_total)
  if pos_total!=0:
    print(pos_image_max,round(pos_image_max/pos_total,4))
  if neg_total!=0:
    print(neg_image_max,round(neg_image_max/neg_total,4))
  print("accuracy",round((pos_image_max+neg_image_max)/(pos_total+neg_total),4))
