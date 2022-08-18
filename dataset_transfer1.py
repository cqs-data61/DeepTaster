
"""
Generate DFT images
"""

import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor 
from torch import nn
from keras.datasets import cifar100
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import foolbox as fb


parser = argparse.ArgumentParser(description='DFT image generation')
parser.add_argument('--model', default='Imagenet', type=str, help='Imagenet for pretrained imagenet model or model path')
parser.add_argument('--architecture', default='Resnet101', type=str, help='model architecture')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset for DFT image ganeration')
parser.add_argument('--type', default='train', type=str, help='type of DFT images. choice: test/val/train')
parser.add_argument('--output', default='./DFTimages', type=str, help='DFT images saved dir')

if __name__ == '__main__':
    opt = parser.parse_args()
    #Load model
    #Load Pretrained imagenet model
    if opt.model == 'Imagenet':
        if opt.architecture == 'Resnet101:
            model=models.resnet101(pretrained=True)
        elif opt.architecture == 'Densenet161':
            model=models.densenet161(pretrained=True)
        elif opt.architecture == 'Vgg16':
            model=models.vgg16(pretrained=True)
    else:
        model=torch.load(opt.model)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


#load dataset
if opt.dataset=='cifar100':
    (X_train, y_train), (X_test1, y_test1) = cifar100.load_data()
    X_test1=X_test1.reshape(10000,3,32,32)
    y_test1=tf.keras.utils.to_categorical(y_test1)
    X_test1=X_test1/255
    X_test1=X_test1.astype(np.float32)

elif opt.dataset=='imagenet':
    (X_train, y_train), (X_test1, y_test1) = imagenet.load_data()
    X_test1=X_test1.reshape(10000,3,224,224)
    y_test1=tf.keras.utils.to_categorical(y_test1)
    X_test1=X_test1/255
    X_test1=X_test1.astype(np.float32)





bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds)
fmodel = fmodel.transform_bounds((0, 1))
for k in range(9):
  X_test=torch.from_numpy(X_test1[0+32*k:32+32*k]).float().to(device)
  X_test=F.interpolate(X_test, size=(224, 224), mode='bicubic', align_corners=False)
  y_test=[]
  for j in range(32):
    y_test.append(torch.argmax(model(X_test)[j]))
  y_test=torch.tensor(y_test).to(device)
  # y_test=np.argmax(y_test1, axis=1)
  # y_test=torch.from_numpy(y_test[0+32*k:32+32*k]).to(device)
  attack = fb.attacks.FGSM()
  model_name='VGG'
  epsilon=0.03
  attackname="FGSM"
  filepath='/content/drive/MyDrive/Watermark_dnn/temp'
  filepath2='/content/drive/MyDrive/Watermark_dnn/Imagenet_test2/test_densenet'
  raw, clipped, is_adv = attack(fmodel,X_test,y_test, epsilons=epsilon)
  for i in range(32):
    plt.figure(num=None, figsize=(4,3), dpi=150)
    plt.figure(figsize = (2,2))
    plt.imshow((clipped[i]-X_test[i]).cpu().permute(1,2,0));
    plt.axis('off')
    plt.savefig('./per'+str(i+32*k)+'.jpg', dpi=150,bbox_inches='tight', pad_inches=0)
  for i in range(32):
    img_c1=cv2.imread(os.path.join(filepath,'per'+str(i+32*k)+'.jpg'), 0)
    img_c2 = np.fft.fft2(img_c1)
    img_c3 = np.fft.fftshift(img_c2)
    cv2.imwrite(os.path.join(filepath2,'densenet'+str(i+32*k)+'.jpg'),20*np.log(1+np.abs(img_c3)))

import os
print(len(os.listdir(filepath2)))

#extraction case

import os
import foolbox as fb
epoch_list=['vgg','densenet','resnet']

for epochsize in epoch_list:
  os.mkdir("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/black_box/plusone_images/"+epochsize)
  model=torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/black_box/plusone_models/"+epochsize+'.pt')
  model=model.to(device)
  bounds = (0, 1)
  fmodel = fb.PyTorchModel(model, bounds=bounds)
  fmodel = fmodel.transform_bounds((0, 1))
  for k in range(9):
    X_test=torch.from_numpy(X_test1[0+32*k:32+32*k]).float().to(device)
    X_test=F.interpolate(X_test, size=(224, 224), mode='bicubic', align_corners=False)
    # y_test=[]
    # for j in range(32):
    #   y_test.append(torch.argmax(model(X_test)[j]))
    # y_test=torch.tensor(y_test).to(device)
    y_test=np.argmax(y_test1, axis=1)
    y_test=torch.from_numpy(y_test[0+32*k:32+32*k]).to(device)
    attack = fb.attacks.FGSM()
    model_name='VGG'
    epsilon=0.03
    attackname="FGSM"
    filepath='/content/drive/MyDrive/Watermark_dnn/temp'
    filepath2="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/black_box/plusone_images/"+epochsize
    raw, clipped, is_adv = attack(fmodel,X_test,y_test, epsilons=epsilon)
    for i in range(32):
      plt.figure(num=None, figsize=(4,3), dpi=150)
      plt.figure(figsize = (2,2))
      plt.imshow((clipped[i]-X_test[i]).cpu().permute(1,2,0));
      plt.axis('off')
      plt.savefig('./per'+str(i+32*k)+'.jpg', dpi=150,bbox_inches='tight', pad_inches=0)
    for i in range(32):
      img_c1=cv2.imread(os.path.join(filepath,'per'+str(i+32*k)+'.jpg'), 0)
      img_c2 = np.fft.fft2(img_c1)
      img_c3 = np.fft.fftshift(img_c2)
      cv2.imwrite(os.path.join(filepath2,'random'+str(i+32*k)+'.jpg'),20*np.log(1+np.abs(img_c3)))

#transfer learning cese

import os
import foolbox as fb
epoch_list=[20,40,60,80,100,120,140,160,180,200]

for epochsize in epoch_list:
  os.mkdir("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_70/images/resnet_"+str(epochsize))
  model=torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_70/models/resnet_"+str(epochsize)+'.pt')
  model=model.to(device)
  bounds = (0, 1)
  fmodel = fb.PyTorchModel(model, bounds=bounds)
  fmodel = fmodel.transform_bounds((0, 1))
  for k in range(9):
    X_test=torch.from_numpy(X_test1[0+32*k:32+32*k]).float().to(device)
    X_test=F.interpolate(X_test, size=(224, 224), mode='bicubic', align_corners=False)
    y_test=[]
    for j in range(32):
      y_test.append(torch.argmax(model(X_test)[j]))
    y_test=torch.tensor(y_test).to(device)
    # y_test=np.argmax(y_test1, axis=1)
    # y_test=torch.from_numpy(y_test[0+32*k:32+32*k]).to(device)
    attack = fb.attacks.FGSM()
    model_name='VGG'
    epsilon=0.03
    attackname="FGSM"
    filepath='/content/drive/MyDrive/Watermark_dnn/temp'
    filepath2="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_70/images/resnet_"+str(epochsize)
    raw, clipped, is_adv = attack(fmodel,X_test,y_test, epsilons=epsilon)
    for i in range(32):
      plt.figure(num=None, figsize=(4,3), dpi=150)
      plt.figure(figsize = (2,2))
      plt.imshow((clipped[i]-X_test[i]).cpu().permute(1,2,0));
      plt.axis('off')
      plt.savefig('./per'+str(i+32*k)+'.jpg', dpi=150,bbox_inches='tight', pad_inches=0)
    for i in range(32):
      img_c1=cv2.imread(os.path.join(filepath,'per'+str(i+32*k)+'.jpg'), 0)
      img_c2 = np.fft.fft2(img_c1)
      img_c3 = np.fft.fftshift(img_c2)
      cv2.imwrite(os.path.join(filepath2,'transfer'+str(i+32*k)+'.jpg'),20*np.log(1+np.abs(img_c3)))

#retraining cese

import os
import foolbox as fb
datasize_list=[10,30,50,70,90]
epoch_list=[50, 100,150,200]
for datasize in datasize_list:
  for epochsize in epoch_list:
    os.mkdir("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_resnet/images/resnet"+str(datasize)+'_'+str(epochsize))
    model=torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_resnet/models/resnet"+str(datasize)+'_'+str(epochsize)+'.pt')
    model=model.to(device)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds)
    fmodel = fmodel.transform_bounds((0, 1))
    for k in range(9):
      X_test=torch.from_numpy(X_test1[0+32*k:32+32*k]).float().to(device)
      X_test=F.interpolate(X_test, size=(224, 224), mode='bicubic', align_corners=False)
      # y_test=[]
      # for j in range(32):
      #   y_test.append(torch.argmax(model(X_test)[j]))
      # y_test=torch.tensor(y_test).to(device)
      y_test=np.argmax(y_test1, axis=1)
      y_test=torch.from_numpy(y_test[0+32*k:32+32*k]).to(device)
      attack = fb.attacks.FGSM()
      model_name='VGG'
      epsilon=0.03
      attackname="FGSM"
      filepath='/content/drive/MyDrive/Watermark_dnn/temp'
      filepath2="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_resnet/images/resnet"+str(datasize)+'_'+str(epochsize)
      raw, clipped, is_adv = attack(fmodel,X_test,y_test, epsilons=epsilon)
      for i in range(32):
        plt.figure(num=None, figsize=(4,3), dpi=150)
        plt.figure(figsize = (2,2))
        plt.imshow((clipped[i]-X_test[i]).cpu().permute(1,2,0));
        plt.axis('off')
        plt.savefig('./per'+str(i+32*k)+'.jpg', dpi=150,bbox_inches='tight', pad_inches=0)
      for i in range(32):
        img_c1=cv2.imread(os.path.join(filepath,'per'+str(i+32*k)+'.jpg'), 0)
        img_c2 = np.fft.fft2(img_c1)
        img_c3 = np.fft.fftshift(img_c2)
        cv2.imwrite(os.path.join(filepath2,'retraining'+str(i+32*k)+'.jpg'),20*np.log(1+np.abs(img_c3)))

#finetuning case

import os
import foolbox as fb
datasize_list=[100,500,1000,2500]
epoch_list=[1,10,20,30,40]
for datasize in datasize_list:
  for epochsize in epoch_list:
    model=torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/finetuning/models/densenet"+str(datasize)+'_'+str(epochsize)+'.pt')
    model=model.to(device)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds)
    fmodel = fmodel.transform_bounds((0, 1))
    for k in range(9):
      X_test=torch.from_numpy(X_test1[0+32*k:32+32*k]).float().to(device)
      X_test=F.interpolate(X_test, size=(224, 224), mode='bicubic', align_corners=False)
      y_test=[]
      for j in range(32):
        y_test.append(torch.argmax(model(X_test)[j]))
      y_test=torch.tensor(y_test).to(device)
      # y_test=np.argmax(y_test1, axis=1)
      # y_test=torch.from_numpy(y_test[0+32*k:32+32*k]).to(device)
      attack = fb.attacks.FGSM()
      model_name='VGG'
      epsilon=0.03
      attackname="FGSM"
      filepath='/content/drive/MyDrive/Watermark_dnn/temp'
      filepath2="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/finetuning/images/finetuning"+str(datasize)+'_'+str(epochsize)
      raw, clipped, is_adv = attack(fmodel,X_test,y_test, epsilons=epsilon)
      for i in range(32):
        plt.figure(num=None, figsize=(4,3), dpi=150)
        plt.figure(figsize = (2,2))
        plt.imshow((clipped[i]-X_test[i]).cpu().permute(1,2,0));
        plt.axis('off')
        plt.savefig('./per'+str(i+32*k)+'.jpg', dpi=150,bbox_inches='tight', pad_inches=0)
      for i in range(32):
        img_c1=cv2.imread(os.path.join(filepath,'per'+str(i+32*k)+'.jpg'), 0)
        img_c2 = np.fft.fft2(img_c1)
        img_c3 = np.fft.fftshift(img_c2)
        cv2.imwrite(os.path.join(filepath2,'squeeze'+str(i+32*k)+'.jpg'),20*np.log(1+np.abs(img_c3)))

#pruning case

import os
import foolbox as fb
datasize_list=['3','5','7','9']
epoch_list=[20,40]
for datasize in datasize_list:
  for epochsize in epoch_list:
    os.mkdir("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_sp/resnet_"+datasize+"/images/resnet_"+str(epochsize))
    model=torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_sp/resnet_"+datasize+'/models/resnet_'+str(epochsize)+'.pt')
    model=model.to(device)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds)
    fmodel = fmodel.transform_bounds((0, 1))
    for k in range(9):
      X_test=torch.from_numpy(X_test1[0+32*k:32+32*k]).float().to(device)
      X_test=F.interpolate(X_test, size=(224, 224), mode='bicubic', align_corners=False)
      y_test=[]
      # for j in range(32):
      #   y_test.append(torch.argmax(model(X_test)[j]))
      # y_test=torch.tensor(y_test).to(device)
      y_test=np.argmax(y_test1, axis=1)
      y_test=torch.from_numpy(y_test[0+32*k:32+32*k]).to(device)
      attack = fb.attacks.FGSM()
      model_name='VGG'
      epsilon=0.03
      attackname="FGSM"
      filepath='/content/drive/MyDrive/Watermark_dnn/temp'
      filepath2="/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_sp/resnet_"+datasize+"/images/resnet_"+str(epochsize)
      raw, clipped, is_adv = attack(fmodel,X_test,y_test, epsilons=epsilon)
      for i in range(32):
        plt.figure(num=None, figsize=(4,3), dpi=150)
        plt.figure(figsize = (2,2))
        plt.imshow((clipped[i]-X_test[i]).cpu().permute(1,2,0));
        plt.axis('off')
        plt.savefig('./per'+str(i+32*k)+'.jpg', dpi=150,bbox_inches='tight', pad_inches=0)
      for i in range(32):
        img_c1=cv2.imread(os.path.join(filepath,'per'+str(i+32*k)+'.jpg'), 0)
        img_c2 = np.fft.fft2(img_c1)
        img_c3 = np.fft.fftshift(img_c2)
        cv2.imwrite(os.path.join(filepath2,'prune'+str(i+32*k)+'.jpg'),20*np.log(1+np.abs(img_c3)))

"""model pruning"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/vgg16.pt")
model=model.to(device)

print(model)

amount=0.6
import torch.nn.utils.prune as prune
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/resnet18.pt")
model=model.to(device)

for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=amount)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=amount)
    elif isinstance(module, torch.nn.BatchNorm2d):
        prune.l1_unstructured(module, name='weight', amount=amount)

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

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

train_len=len(train_data)

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

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

torch.save(model, "/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/pruning_finetuning/models3/resnet_60.pt")

"""Resnet training"""

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=False)
# class_num=100
# fc=model.fc
# in_dim = fc.in_features
# model.fc=nn.Linear(in_dim,class_num)
# model=model.to(device)
# model=torch.load("/content/drive/MyDrive/Watermark_dnn/CIFAR_model_224input/resnet18.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
#model=model.to(device)

class_num=10
fc=model.fc
in_dim = fc.in_features
model.fc=nn.Linear(in_dim,class_num)

#print(model)

# num_ftrs = model.classifier[0].in_features
# model.classifier[0] = nn.Linear(num_ftrs, 100)

model=model.to(device)

from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# transform = transforms.Compose(
#     [
#      transforms.ToTensor(),
#      transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train_data = datasets.MNIST(
#     root = 'data',
#     train=True,                      
#     transform = transform, 
#     download = True,            
# )
# test_data = datasets.MNIST(
#     root = 'data',
#     train=False,                                 
#     transform = transform, 
#     download = True,            
# )

# batch_size = 128
# train_loader = torch.utils.data.DataLoader(
#                  dataset=train_data,
#                  batch_size=batch_size,
#                  shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#                 dataset=test_data,
#                 batch_size=batch_size,
#                 shuffle=True)

##Get CIFAR100 dataset

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

train_len=len(train_data)
datasize=10
train_size=5000
train_set, val_set = torch.utils.data.random_split(train_data, [train_size, train_len-train_size])
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

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)

torch.save(resnet18,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/cifar10_training/models/resnet.pt')





"""Transfer learning

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
  if ct < 5:
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
          if ct < 5:
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
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_20.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_40.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_60.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_80.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_100.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_120.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_140.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_160.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_180.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
torch.save(model,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/transfer_office_freezing/freeze_50/models/resnet_200.pt')

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

