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
