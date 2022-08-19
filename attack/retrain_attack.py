
"""

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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.squeezenet1_0(pretrained=True)
model=model.to(device)

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
datasize=10
train_size=5000

batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_data,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=True)

from sklearn.model_selection import StratifiedShuffleSplit

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
datasize=0.9
train_size=5000

sss = StratifiedShuffleSplit(n_splits=1, test_size=datasize, random_state=1)
indices = list(range(len(train_data)))
y_test0 = [y for _, y in train_data]

for test_index, val_index in sss.split(indices, y_test0):
    print('test:', test_index, 'val:', val_index)
    print(len(val_index), len(test_index))

from torch.utils.data import Subset

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

#model=torch.load("/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_uniform_resnet/models/resnet_10_50.pt")
#model=model.to(device)
resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(resnet18,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_uniform_resnet/models/resnet100_50.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(resnet18,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_uniform_resnet/models/resnet100_100.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(resnet18,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_uniform_resnet/models/resnet100_150.pt')

resnet18 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(resnet18,  '/content/drive/MyDrive/Watermark_dnn/cifar_dataset_experiment/retraining_uniform_resnet/models/resnet100_200.pt')
