
"""
Retraining attack
"""
import argparse
import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor 
from torch import nn
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import copy
import time
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='DFT image generation')
parser.add_argument('--dataset', required=True, type=str, help='train dataset: cifar100, Imagenet')
parser.add_argument('--imagenerpath', default="./", type=str, help='dataset path')
parser.add_argument('--architecture', required=True, type=str, help='train architecture: Resnet101, Vgg16, Densenet161')
parser.add_argument('--output', required=True, type=str, help='output images saved dir')
parser.add_argument('--random_seed', default=80, type=int, help='model initializaion random seed')

#train function
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
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
            inputs, labels = data
            

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / len(train_data)
        avg_acc = acc_train / len(train_data)
        
        
        print()
        # * 2 as we only used half of the dataset
        
        model.train(False)
        model.eval()
            
        for i, data in enumerate(test_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            

            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)

            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(test_data)
        avg_acc_val = acc_val / len(test_data)
        
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
            best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

#load model architecture
torch.manual_seed(opt.random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opt.dataset=='Imagenet':
    if opt.architecture=='Resnet101':
        model = models.resnet101(pretrained=False)
    elif opt.architecture=='Vgg16':
        model = models.vgg16(pretrained=False)
    elif opt.architecture=='Densenet161':
        model = models.densenet161(pretrained=False)
    model=model.to(device)
elif opt.dataset=='cifar100':
    if opt.architecture=='Resnet101':
        model = models.resnet101(pretrained=False)
        class_num=100
        fc=model.fc
        in_dim = fc.in_features
        model.fc=nn.Linear(in_dim,class_num)
    elif opt.architecture=='Vgg16':
        model = models.vgg16(pretrained=False)
        for param in vgg16.features.parameters():
            param.require_grad = False
        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1] 
        features.extend([nn.Linear(num_features,100)])
        vgg16.classifier = nn.Sequential(*features) 
    elif opt.architecture=='Densenet161':
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 100)
    model=model.to(device)
else:
    print("dataset error")
    exit
    
#setting train hyper-parameter
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


#load dataset
if opt.dataset=='cifar100':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR100(
        root = 'data',
        train = True,                         
        transform = transform, 
        download = True,            
    )
    test_data = datasets.CIFAR100(
        root = 'data', 
        train = False, 
        transform = transform
    )
else if opt.dataset=='Imagenet':
    #load imagenet dataset
    train_data = torchvision.datasets.ImageNet(opt.imagenetpath, split='train', download=None, transform=transform)
    test_data = torchvision.datasets.ImageNet(opt.imagenetpath, split='val', download=None, transform=transform)

#class uniform split      
for size in [10,30,50,70,80,90,100]:
    train_len=len(train_data)
    datasize=size/100
    sss = StratifiedShuffleSplit(n_splits=1, test_size=datasize, random_state=opt.random_seed)
    indices = list(range(len(train_data)))
    y_test0 = [y for _, y in train_data]

    for test_index, val_index in sss.split(indices, y_test0):
        print('test:', test_index, 'val:', val_index)
        print(len(val_index), len(test_index))

    val_ds = Subset(train_data, val_index)
    test_ds = Subset(train_data, test_index)
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
                     dataset=val_ds,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_data,
                    batch_size=batch_size,
                    shuffle=True)
   
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
    torch.save(model,  opt.output+'/'+architecture+str(size)+'_50.pt')

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
    torch.save(model,  opt.output+'/'+architecture+str(size)+'_100.pt')

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
    torch.save(model,  opt.output+'/'+architecture+str(size)+'_150.pt')

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
    torch.save(model,  opt.output+'/'+architecture+str(size)+'_200.pt')
