
"""
Detection Classifier Generation
"""

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

parser = argparse.ArgumentParser(description='DFT image generation')
parser.add_argument('--train', required=True, type=str, help='train dataset path')
# parser.add_argument('--test', default='Resnet101', type=str, help='test dataset path')
# parser.add_argument('--val', default='cifar100', type=str, help='validation dataset path')
parser.add_argument('--output', default='./', type=str, help='detection classifier saved dir')

if __name__ == '__main__':
dataset = dset.ImageFolder(root="/content/drive/MyDrive/Watermark_dnn/Imagenet_train2",
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

import torchvision.datasets as dset
dataset = dset.ImageFolder(root="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar_test_224",
                           transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.Resize(28),      
                               transforms.CenterCrop(28), 
                               transforms.ToTensor(),    
                           ]))

test_dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=16,
                                         shuffle=False,
                                         num_workers=8)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({
       'num_epochs':10,
       'num_epochs_ae':40,
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

deep_SVDD = TrainerDeepSVDD(args, train_dataloader, device)
if args.pretrain:
     deep_SVDD.pretrain()

net = DeepSVDD_network().to(device)
        

state_dict = torch.load('/content/drive/MyDrive/Watermark_dnn/deep_one_class/pretrained_parameters.pth')
net.load_state_dict(state_dict['net_dict'])
c = torch.Tensor(state_dict['center']).to(device)


optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=args.lr_milestones, gamma=0.1)

net.train()
for epoch in range(2):
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

labels,scores=eval(net,c,test_dataloader,device)

import matplotlib.pyplot as plt
unless=[0]*10
for i in range(len(labels)):
  if labels[i]==0:
    if unless[0]==1: 
      plt.plot(i,scores[i],'.', color='black')
    else:
      plt.plot(i,scores[i],'.', color='black',label="cifar_densenet")
      unless[0]=1
  elif labels[i]==1:
    if unless[1]==1: 
      plt.plot(i,scores[i],'.', color='r')
    else:
      plt.plot(i,scores[i],'.', color='r',label="cifar_resnet")
      unless[1]=1
  elif labels[i]==2:
    if unless[2]==1: 
      plt.plot(i,scores[i],'.', color='g')
    else:
      plt.plot(i,scores[i],'.', color='g',label="cifar_vgg")
      unless[2]=1
  elif labels[i]==3:
    if unless[3]==1: 
      plt.plot(i,scores[i],'.', color='blue')
    else:
      plt.plot(i,scores[i],'.', color='blue',label="imagenet_densenet")
      unless[3]=1
  elif labels[i]==4:
    if unless[4]==1: 
      plt.plot(i,scores[i],'.', color='blue')
    else:
      plt.plot(i,scores[i],'.', color='blue',label="imagenet_resnet")
      unless[4]=1
  elif labels[i]==5:
    if unless[5]==1: 
      plt.plot(i,scores[i],'.', color='blue')
    else:
      plt.plot(i,scores[i],'.', color='blue',label="imagenet_vgg")
      unless[5]=1
  elif labels[i]==6:
    if unless[6]==1: 
      plt.plot(i,scores[i],'.', color='orange')
    else:
      plt.plot(i,scores[i],'.', color='orange',label="val_densenet")
      unless[6]=1
  elif labels[i]==7:
    if unless[7]==1: 
      plt.plot(i,scores[i],'.', color='orange')
    else:
      plt.plot(i,scores[i],'.', color='orange',label="val_resnet")
      unless[7]=1
  elif labels[i]==8:
    if unless[8]==1: 
      plt.plot(i,scores[i],'.', color='orange')
    else:
      plt.plot(i,scores[i],'.', color='orange',label="val_vgg")
      unless[8]=1
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()

import numpy

scores = []
net.eval()
print('Testing...')
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.float().to(device)
        z = net(x)
        score = torch.sum((z - c) ** 2, dim=1)
        for i in range(16):
          if y[i]==6 or y[i]==7 or y[i]==8:
            scores.append(score[i])
print(numpy.mean(scores),numpy.std(scores))

import math
scores.sort()
threshold=scores[855]
print(threshold)

scores = []
net.eval()
print('Testing...')
image_max=0
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.float().to(device)
        z = net(x)
        score = torch.sum((z - c) ** 2, dim=1)
        for i in range(16):
          if y[i]==0 or y[i]==1 or y[i]==2:
            if score[i]<=threshold:
              image_max+=1

print(image_max,round(image_max/864,4))

scores = []
net.eval()
print('Testing...')
image_max=0
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.float().to(device)
        z = net(x)
        score = torch.sum((z - c) ** 2, dim=1)
        for i in range(16):
          if y[i]==3 or y[i]==4 or y[i]==5:
            if score[i]>threshold:
              image_max+=1

print(image_max,round(image_max/864,4))











torch.save(net,  '/content/drive/MyDrive/Watermark_dnn/deep_one_class/Imagenet5.pt')



!rm -r /content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/office_scratch/images2/resnet_200



#test 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('/content/drive/MyDrive/Watermark_dnn/deep_one_class/cifar_1600_1_parameter.pth')
c1 = torch.Tensor(state_dict['center']).to(device)
net1= torch.load('/content/drive/MyDrive/Watermark_dnn/deep_one_class/cifar_1600_1.pt')

import torchvision.datasets as dset
dataset = dset.ImageFolder(root="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar_test_224",
                           transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.Resize(28),      
                               transforms.CenterCrop(28), 
                               transforms.ToTensor(),    
                           ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=16,
                                         shuffle=False,
                                         num_workers=8)

#.평균 표편 구하기
import numpy

scores = []
net1.eval()
print('Testing...')
with torch.no_grad():
    for x, y in dataloader:
        for i in range(16):
          if y[i]==8 or y[i]==9 or y[i]==10:      
            x = x.float().to(device)
            z = net1(x)
            score = torch.sum((z - c1) ** 2, dim=1)
            scores.append(score[i])
print(numpy.mean(scores),numpy.std(scores))

#threshold 구하기
scores.sort()
print(scores[855]*1000)
print(len(scores))



#threshold 구하기
import math
threshold=scores[855]
#threshold=scores[570]

print(threshold)



import torchvision.datasets as dset
dataset = dset.ImageFolder(root="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_sp/resnet_9/images",
                           transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.Resize(28),      
                               transforms.CenterCrop(28), 
                               transforms.ToTensor(),    
                           ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=16,
                                         shuffle=False,
                                         num_workers=8)

#accuracy 구하기
for k in range(3):
  scores = []
  net1.eval()
  print('Testing...')
  image_max=0
  with torch.no_grad():
      for x, y in dataloader:          
          for i in range(16):
            if y[i]==k:
              x = x.float().to(device)
              z = net1(x)
              score = torch.sum((z - c1) ** 2, dim=1)
              if score[i]<=threshold:
                image_max+=1

  print(k, ": ",image_max,round(image_max/288,4))

#최소최대 구하기
scores = []
net1.eval()
print('Testing...')
image_max=0
image_min=1
with torch.no_grad():
    for x, y in dataloader:
        x = x.float().to(device)
        z = net1(x)
        score = torch.sum((z - c1) ** 2, dim=1)
        for i in range(16):
          if y[i]==8:
            if score[i]>image_max:
              image_max=score[i]
            if score[i]<image_min:
              image_min=score[i]
        scores.append(score.detach().cpu())
print(image_max,image_min)

labels,scores=eval(net1,c1,dataloader,device)

#imagenet
import matplotlib.pyplot as plt
unless=[0]*9
for i in range(len(labels)):
  if labels[i]==0:
    if unless[0]==1: 
      plt.plot(i,scores[i],'.', color='black')
    else:
      plt.plot(i,scores[i],'.', color='black',label="Imagenet_resnet")
      unless[0]=1
  elif labels[i]==1:
    if unless[1]==1: 
      plt.plot(i,scores[i],'.', color='r')
    else:
      plt.plot(i,scores[i],'.', color='r',label="Imagenet_dense")
      unless[1]=1
  elif labels[i]==2:
    if unless[2]==1: 
      plt.plot(i,scores[i],'.', color='g')
    else:
      plt.plot(i,scores[i],'.', color='g',label="Imagenet_google")
      unless[2]=1
  elif labels[i]==3:
    if unless[3]==1: 
      plt.plot(i,scores[i],'.', color='y')
    else:
      plt.plot(i,scores[i],'.', color='y',label="Imagenet_vgg")
      unless[3]=1
  elif labels[i]==4:
    if unless[4]==1: 
      plt.plot(i,scores[i],'.', color='blue')
    else:
      plt.plot(i,scores[i],'.', color='blue',label="resnet_car")
      unless[4]=1
  elif labels[i]==5:
    if unless[5]==1: 
      plt.plot(i,scores[i],'.', color='orange')
    else:
      plt.plot(i,scores[i],'.', color='orange',label="resnet_CIFAR_25")
      unless[5]=1
  elif labels[i]==6:
    if unless[6]==1: 
      plt.plot(i,scores[i],'.', color='purple')
    else:
      plt.plot(i,scores[i],'.', color='purple',label="resnet_CIFAR_50")
      unless[6]=1
  elif labels[i]==7:
    if unless[7]==1: 
      plt.plot(i,scores[i],'.', color='grey')
    else:
      plt.plot(i,scores[i],'.', color='grey',label="resnet_CIFAR_75")
      unless[7]=1
  elif labels[i]==8:
    if unless[8]==1: 
      plt.plot(i,scores[i],'.', color='indigo')
    else:
      plt.plot(i,scores[i],'.', color='indigo',label="resnet_cifar")
      unless[8]=1
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()

labels,scores=eval(net,c,test_dataloader,device)

#cifar
import matplotlib.pyplot as plt
unless=[0]*9
for i in range(len(labels)):
  if labels[i]==0:
    if unless[0]==1: 
      plt.plot(i,scores[i],'.', color='r')
    else:
      plt.plot(i,scores[i],'.', color='r',label="cifar_inception")
      unless[0]=1
  elif labels[i]==1:
    if unless[1]==1: 
      plt.plot(i,scores[i],'.', color='r')
    else:
      plt.plot(i,scores[i],'.', color='r',label="cifar_resnet")
      unless[1]=1
  elif labels[i]==2:
    if unless[2]==1: 
      plt.plot(i,scores[i],'.', color='b')
    else:
      plt.plot(i,scores[i],'.', color='b',label="cifar_vgg")
      unless[2]=1
  elif labels[i]==3:
    if unless[3]==1: 
      plt.plot(i,scores[i],'.', color='r')
    else:
      plt.plot(i,scores[i],'.', color='r',label="10_10")
      unless[3]=1
  # elif labels[i]==4:
  #   if unless[4]==1: 
  #     plt.plot(i,scores[i],'.', color='blue')
  #   else:
  #     plt.plot(i,scores[i],'.', color='blue',label="resnet_car")
  #     unless[4]=1
  # elif labels[i]==5:
  #   if unless[5]==1: 
  #     plt.plot(i,scores[i],'.', color='b')
  #   else:
  #     plt.plot(i,scores[i],'.', color='b',label="resnet_CIFAR_25")
  #     unless[5]=1
  # elif labels[i]==6:
  #   if unless[6]==1: 
  #     plt.plot(i,scores[i],'.', color='y')
  #   else:
  #     plt.plot(i,scores[i],'.', color='y',label="resnet_CIFAR_50")
  #     unless[6]=1
  # elif labels[i]==7:
  #   if unless[7]==1: 
  #     plt.plot(i,scores[i],'.', color='black')
  #   else:
  #     plt.plot(i,scores[i],'.', color='black',label="resnet_CIFAR_75")
  #     unless[7]=1
  # elif labels[i]==8:
  #   if unless[8]==1: 
  #     plt.plot(i,scores[i],'.', color='black')
  #   else:
  #     plt.plot(i,scores[i],'.', color='black',label="resnet_cifar")
  #     unless[8]=1
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()



print(np.std([98.61,99.65,95.49]))

