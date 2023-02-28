"""
Detection Classifier Generation
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
parser.add_argument('--dataset', default='./DFTimages', type=str, help='dataset path')
parser.add_argument('--preepochs', default=40, type=int, help='autoencoder training epochs')
parser.add_argument('--epochs', default=5, type=int, help='detection classifier training epochs')
parser.add_argument('--output',default='./Deepsvdd', type=str, help='detection classifier saved dir')

if __name__ == '__main__':

  opt = parser.parse_args()
  if not os.path.exists(opt.output):
    os.mkdir(opt.output)
  dataset = dset.ImageFolder(root=opt.dataset+'/train',
                             transform=transforms.Compose([
                                 transforms.Grayscale(),
                                 transforms.Resize(28),      
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor(),    
                             ]))

  train_dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=8)

  
  dataset = dset.ImageFolder(root=opt.dataset+'/val',
                             transform=transforms.Compose([
                                 transforms.Grayscale(),
                                 transforms.Resize(28),      
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor(),    
                             ]))

  val_dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
                                           shuffle=False,
                                           num_workers=8)


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  args = easydict.EasyDict({
         'num_epochs':opt.epochs,
         'num_epochs_ae':opt.preepochs,
         'lr':1e-3,
         'lr_ae':1e-3,
         'weight_decay':5e-7,
         'weight_decay_ae':5e-3,
         'lr_milestones':[50],
         'batch_size':1024,
         'pretrain':True,
         'latent_dim':32,
         'normal_class':0
                  })

  deep_SVDD = TrainerDeepSVDD(args, train_dataloader, device, opt.output+'/pretrained.pth')
  if args.pretrain:
       deep_SVDD.pretrain()

  net = DeepSVDD_network().to(device)


  state_dict = torch.load(opt.output+'/pretrained.pth')
  net.load_state_dict(state_dict['net_dict'])
  c = torch.Tensor(state_dict['center']).to(device)


  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
              milestones=args.lr_milestones, gamma=0.1)

  net.train()
  for epoch in range(opt.epochs):
      total_loss = 0
      for x, _ in train_dataloader:
          x = x.float().to(device)

          optimizer.zero_grad()
          z = net(x)
          loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
          loss.backward()
          optimizer.step()

          total_loss += loss.item()
      scheduler.step()
      print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
              epoch+1, total_loss/len(train_dataloader)))

  scores = []
  net.eval()
  print('Testing...')
  with torch.no_grad():
      for x, y in val_dataloader:
          x = x.float().to(device)
          z = net(x)
          score = torch.sum((z - c) ** 2, dim=1).cpu()
          for i in range(16):
            scores.append(score[i])
  scores.sort()
  threshold=scores[round(len(scores)*0.96)]
  print("threshold: ", threshold.to(torch.float))
  threshold_file=open(opt.output+'/threshold.txt','w')
  threshold_file.write(str(float(threshold)))
  threshold_file.close()
  torch.save(net, opt.output+'/deepsvdd.pt') 
