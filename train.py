import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor 
from torch import nn
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torch._utils import _accumulate
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import copy
import time
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--dataset', required=True, type=str, help='train dataset: cifar10, tiny_imagenet, MNIST')
parser.add_argument('--architecture', required=True, type=str, help='train architecture: Resnet18, Vgg16, Densenet161, Alexnet')
parser.add_argument('--epochs',  default=50, type=int, help='train epochs')
opt = parser.parse_args()

#set gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get tiny imagenet
def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)
    kwargs = {}
    return dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#set random seed number for victim models
torch.manual_seed(107)

# Set dataset
if opt.dataset=='tiny_imagenet':
    class_num=100
    if not os.path.exists("./tiny_imagenet"):
        os.mkdir("./tiny_imagenet")
    if not os.path.exists("./tiny_imagenet/model"):
        os.mkdir("./tiny_imagenet/model")
    DATA_DIR = './tiny-imagenet-200' # Original images come in shapes of [3,64,64]
    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    train_data = generate_dataloader(TRAIN_DIR, "train",
                                    transform=transform)
    test_data=generate_dataloader(VALID_DIR, "val",
                                    transform=transform)

elif opt.dataset=='cifar10': 
    class_num=10
    if not os.path.exists("./cifar10"):
        os.mkdir("./cifar10")
    if not os.path.exists("./cifar10/model"):
        os.mkdir("./cifar10/model")
    train_data = datasets.CIFAR10(
    root = 'data',
    train = True,                         
    transform = transform, 
    download = True,            
    )
    test_data = datasets.CIFAR10(
        root = 'data', 
        train = False, 
        transform = transform
    )
elif opt.dataset=='MNIST':
    class_num=10
    if not os.path.exists("./MNIST"):
        os.mkdir("./MNIST")
    if not os.path.exists("./MNIST/model"):
        os.mkdir("./MNIST/model")
    transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.CenterCrop(224),
     transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.MNIST(
        root = 'data',
        train=True,                      
        transform = transform, 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data',
        train=False,                                 
        transform = transform, 
        download = True,            
    )
else:
    print("dataset error")
    exit()



batch_size = 128
train_loader = torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=batch_size,
                shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=False)



# Set model
if opt.architecture=='Resnet18':
    model = models.resnet18(pretrained=False)
    fc=model.fc
    in_dim = fc.in_features
    model.fc=nn.Linear(in_dim,class_num)
elif opt.architecture=='Vgg16':
    model = models.vgg16(pretrained=False)
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features,class_num)]) # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier
elif opt.architecture=='Alexnet':
    model = models.alexnet(pretrained=False)
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features,class_num)]) # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier
elif opt.architecture=='Densenet161':
    model = models.densenet161(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, class_num)
else:
    print("architecture error")
    exit()
model=model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


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
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)

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
        if epoch%5==0 or epoch==opt.epochs:
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

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.epochs)
torch.save(resnet18,  './'+opt.dataset+'/model/'+opt.architecture+'_'+str(opt.epochs)+'.pt')
