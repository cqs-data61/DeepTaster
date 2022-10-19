!pip3 install foolbox==3.1.1

import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor 
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(107)
model = models.resnet18(pretrained=False)
class_num=100
fc=model.fc
in_dim = fc.in_features
model.fc=nn.Linear(in_dim,class_num)
model=model.to(device)


from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#get tiny_imagenet dataset
#for a first time
# %cd /content/drive/MyDrive/Watermark_dnn/tiny_imagenet
# !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  
# # Unzip raw zip file
# !unzip -qq 'tiny-imagenet-200.zip'



# tiny imagenet
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets
import os
from torch._utils import _accumulate
from sklearn.model_selection import train_test_split
import numpy as np
def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)


    kwargs = {}
    return dataset
    # dataloader = DataLoader(dataset, batch_size=128, 
    #                     shuffle=(name=="train"), 
    #                     **kwargs)
    
    # return dataloader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DATA_DIR = '~/tiny_imagenet_200' # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')
train_data = generate_dataloader(TRAIN_DIR, "train",
                                  transform=transform)

test_data=generate_dataloader(VALID_DIR, "val",
                                 transform=transform)


batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_data,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=True)



import copy
import time
from torch.autograd import Variable
def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(train_loader)
    val_batches = len(test_loader)
    
    for epoch in range(num_epochs):
        #print("Epoch {}/{}".format(epoch, num_epochs))
        #print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            # if i >= train_batches / 2:
            #     break
        # for i, data in enumerate(train_loader):
        #     if i % 100 == 0:
        #         print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                  
            inputs, labels = data

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / len(train_loader.dataset)
        avg_acc = acc_train / len(train_loader.dataset)
        
        
        print()
        # * 2 as we only used half of the dataset
        
        vgg.train(False)
        vgg.eval()
            
        for i, data in enumerate(test_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
            
            inputs, labels = data
            

            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)

            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(test_loader.dataset)
        avg_acc_val = acc_val / len(test_loader.dataset)
        if epoch%5==0:
          print()
          print("Epoch {} result: ".format(epoch))
          print("Avg loss (train): {:.4f}".format(avg_loss))
          print("Avg acc (train): {:.4f}".format(avg_acc))
          print("Avg loss (val): {:.4f}".format(avg_loss_val))
          print("Avg acc (val): {:.4f}".format(avg_acc_val))
          print('-' * 10)
          print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    #vgg.load_state_dict(best_model_wts)

    return vgg



resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(resnet18,  '~/model/resent20.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(resnet18,  '~/model/resent40.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(resnet18,  '~/model/resent60.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(resnet18,  '~/model/resent80.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(resnet18,  '~/model/resent100.pt')

