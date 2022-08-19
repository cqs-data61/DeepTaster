
"""
Fine Tuning Attack
"""
import argparse


parser = argparse.ArgumentParser(description='DFT image generation')
parser.add_argument('--train', required=True, type=str, help='train dataset path')
# parser.add_argument('--test', default='Resnet101', type=str, help='test dataset path')
# parser.add_argument('--val', default='cifar100', type=str, help='validation dataset path')
parser.add_argument('--output', default='./', type=str, help='detection classifier saved dir')
opt = parser.parse_args()

"""
Transfer learning
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = models.resnet18(pretrained=False)
#model = torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/models/resnet.pt")

# class_num=65
# fc=model.fc
# in_dim = fc.in_features
# model.fc=nn.Linear(in_dim,class_num)

model = torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/resnet18.pt")

#torch.manual_seed(80)
#model=models.resnet18(pretrained=False)
class_num=65
fc=model.fc
in_dim = fc.in_features
model.fc=nn.Linear(in_dim,class_num)
model=model.to(device)

ct = 0
for child in model.children():
  ct += 1
  if ct < 3:
      for param in child.parameters():
          param.requires_grad = False

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.009, momentum=0.95, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

!pip install hub 
!pip install --upgrade urllib3

#from urllib3.util.ssl_ import PROTOCOL_TLS
import hub
ds = hub.load('hub://activeloop/office-home-domain-adaptation')

tform = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5, 0.5, 0.5],
                                                         std = [0.5, 0.5, 0.5])
                           ,transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))])

train_loader = ds.pytorch(num_workers=0, batch_size=128, transform={
                        'images': tform, 'domain_objects':None }, shuffle=True)

print(len(train_loader.dataset))

import copy
import time
from torch.autograd import Variable
import torch.nn.functional as F
def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    
    train_batches = len(train_loader)
    #val_batches = len(test_loader)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        ct = 0
        for child in model.children():
          ct += 1
          if ct < 3:
              for param in child.parameters():
                  param.requires_grad = False
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
            inputs = data['images']
            labels = data['domain_objects']
            labels= torch.max(labels, 1)[0]
            #labels = F.one_hot(labels, num_classes=65)
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
  

        avg_loss = loss_train / 15588
        avg_acc = acc_train / 15588
        
        
        print()
        # * 2 as we only used half of the dataset
        
        vgg.train(False)
        vgg.eval()
            
        # for i, data in enumerate(test_loader):
        #     if i % 100 == 0:
        #         print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
        #     inputs, labels = data
            

        #     inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)

            
        #     optimizer.zero_grad()
            
        #     outputs = vgg(inputs)
            
        #     _, preds = torch.max(outputs.data, 1)
        #     loss = criterion(outputs, labels)
            
        #     loss_val += loss.item()
        #     acc_val += torch.sum(preds == labels.data)
            
        #     del inputs, labels, outputs, preds
        #     torch.cuda.empty_cache()
        
        # avg_loss_val = loss_val / len(test_loader)
        # avg_acc_val = acc_val / len(test_loader)
        
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
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_20.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_40.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_60.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_80.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_100.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_120.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_140.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_160.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_180.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_30/models/resnet_200.pt')

"""Data augmentation"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/models/resnet.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)

class_num=15
fc=model.fc
in_dim = fc.in_features
model.fc=nn.Linear(in_dim,class_num)


model=model.to(device)

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.009, momentum=0.95, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data1 = datasets.CIFAR10(
    root = 'data',
    train = True,                         
    transform = transform, 
    download = True,            
)
test_data1 = datasets.CIFAR10(
    root = 'data', 
    train = False, 
    transform = transform
)
train_data2 = datasets.CIFAR100(
    root = 'data',
    train = True,                         
    transform = transform, 
    download = True,            
)
test_data2 = datasets.CIFAR100(
    root = 'data', 
    train = False, 
    transform = transform
)


label_list=[0,8,16,70,22]
for i in range(50000):
  if train_data2.targets[i] not in label_list:
    train_data2.targets[i]=100
  if train_data2.targets[i]==0:
    train_data2.targets[i]=10
  elif train_data2.targets[i]==8:
    train_data2.targets[i]=11
  elif train_data2.targets[i]==16:
    train_data2.targets[i]=12
  elif train_data2.targets[i]==70:
    train_data2.targets[i]=13
  elif train_data2.targets[i]==22:
    train_data2.targets[i]=14

label_list=[0,8,16,70,22]
for i in range(10000):
  if test_data2.targets[i] not in label_list:
    test_data2.targets[i]=100
  if test_data2.targets[i]==0:
    test_data2.targets[i]=10
  elif test_data2.targets[i]==8:
    test_data2.targets[i]=11
  elif test_data2.targets[i]==16:
    test_data2.targets[i]=12
  elif test_data2.targets[i]==70:
    test_data2.targets[i]=13
  elif test_data2.targets[i]==22:
    test_data2.targets[i]=14

import numpy as np
k=0
for i in range(50000):
  if train_data2.targets[i]!=100:
    if k==0:
      sub_data=train_data2.data[i,:,:,:]
      sub_data=sub_data.reshape(1,32,32,3)
      k=1
    else:
      sub_data=np.append(sub_data,train_data2.data[i,:,:,:].reshape(1,32,32,3),axis=0)
sub_target=[]
for i in range(50000):
  if train_data2.targets[i]!=100:
    sub_target.append(train_data2.targets[i])
train_data2.data=sub_data
train_data2.targets=sub_target

import numpy as np
k=0
for i in range(10000):
  if test_data2.targets[i]!=100:
    if k==0:
      sub_data=test_data2.data[i,:,:,:]
      sub_data=sub_data.reshape(1,32,32,3)
      k=1
    else:
      sub_data=np.append(sub_data,test_data2.data[i,:,:,:].reshape(1,32,32,3),axis=0)
sub_target=[]
for i in range(10000):
  if test_data2.targets[i]!=100:
    sub_target.append(test_data2.targets[i])
test_data2.data=sub_data
test_data2.targets=sub_target

print(train_data2.targets)
print(len(train_data2.targets))
print(test_data2.targets)
print(len(test_data2.targets))

print(sub_data.shape)

train_data = torch.utils.data.ConcatDataset([train_data1, train_data2])
test_data = torch.utils.data.ConcatDataset([test_data1, test_data2])

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
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
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

        avg_loss = loss_train / len(train_data)
        avg_acc = acc_train / len(train_data)
        
        
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
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/from_scratch_models/resnet_transfer_20.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/from_scratch_models/resnet_transfer_40.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/from_scratch_models/resnet_transfer_60.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/from_scratch_models/resnet_transfer_80.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/from_scratch_models/resnet_transfer_100.pt')

"""model extraction"""

#imagenet model extraction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
class_num=100
fc=model.fc
in_dim = fc.in_features
model.fc=nn.Linear(in_dim,class_num)

model=model.to(device)

# model1 = models.vgg16(pretrained=True)
# model1=model1.to(device)

ct = 0
for child in model.children():
  ct += 1
  if ct < 9:
      for param in child.parameters():
          param.requires_grad = False

print(ct)

#CIFAR100 model extraction


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = models.densenet161(pretrained=False)
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, 100)
# model=model.to(device)

model1 = torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/resnet18.pt")
model1=model1.to(device)

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.95, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

import random
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
# for i in range(50000):
#   if train_data.targets[i]!=100:
#     train_data.targets[i]=train_data.targets[i]+1
#   else:
#     train_data.targets[i]==0

# for i in range(10000):
#   if test_data.targets[i]!=100:
#     test_data.targets[i]=test_data.targets[i]+1
#   else:
#     test_data.targets[i]==0
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
def train_model(sub_model,vgg, criterion, optimizer, scheduler, num_epochs=10):
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
        
        vgg.train(True)
        ct = 0
        for child in model.children():
          ct += 1
          if ct < 9:
              for param in child.parameters():
                  param.requires_grad = False
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
            inputs, labels = data
            
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            outputs1=sub_model(inputs)

            _, preds = torch.max(outputs.data, 1)
            _, labels = torch.max(outputs1.data, 1)
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
        
        vgg.train(False)
        vgg.eval()
            
        for i, data in enumerate(test_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            

            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)

            
            optimizer.zero_grad()

            outputs = vgg(inputs)
            outputs1=sub_model(inputs)

            _, preds = torch.max(outputs.data, 1)
            _, labels = torch.max(outputs1.data, 1)
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
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg

model = train_model(model1,model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_sp/resnet_9/models/resnet_60.pt')

"""VGG training

"""



"""https://www.kaggle.com/code/carloalbertobarbano/vgg16-transfer-learning-pytorch/notebook"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=False)

for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features,10)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
model=vgg16.to(device)

#model=torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/vgg16.pt")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_data,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=True)

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.95, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

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
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
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

        avg_loss = loss_train / len(train_data)
        avg_acc = acc_train / len(train_data)
        
        
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
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg

vgg16 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)







torch.save(vgg16,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/models/vgg16.pt')

#Imagenet partial leaked

"""Densenet training"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.densenet161(pretrained=False)

from torch.optim.optimizer import Optimizer, required
import math


num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 10)
# model.features.conv0.apply(squeeze_weights)


model = model.to(device)
#model=torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/densenet1.pt")

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
train_len=len(train_data)

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
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
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

        avg_loss = loss_train / len(train_data)
        avg_acc = acc_train / len(train_data)
        
        
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
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg

densenet = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)

torch.save(densenet,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/models/densenet.pt')





"""Squeezenet training"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.squeezenet1_0(pretrained=False)

model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1,1), stride=(1,1))
model=model.to(device)



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

batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_data,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=True)

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

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
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
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

        avg_loss = loss_train / len(train_loader)
        avg_acc = acc_train / len(train_loader)
        
        
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
        
        avg_loss_val = loss_val / len(test_loader)
        avg_acc_val = acc_val / len(test_loader)
        
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
    
    vgg.load_state_dict(best_model_wts)
    return vgg



squeezenet = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)

torch.save(squeezenet,  '/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/squeezenet1.pt')

