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
