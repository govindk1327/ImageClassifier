import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import random
import os
import numpy as np
from tensorflow import Variable, convert_to_tensor
from torch import unsqueeze
from PIL import Image

parser = argparse.ArgumentParser(description='This is a model training program for a dateset of flowers using pytorch',usage='''
        python train.py (data set shall be initially extracted to the 'flowers' directory)
        python train.py data_dir (data set shall be initially extracted to the 'data_dir' directory)
        python train.py data_dir --save_dir save_directory (set directory to save checkpoints)
        python train.py data_dir --arch "vgg16" (choose architecture from vgg16 and densenet201)
        python train.py data_dir --learning_rate 0.0007 --hidden_units [1920, 710, 200] --epochs 20 (set hyperparameters)''',prog='train')

parser.add_argument('data_directory', action="store", nargs='?', default="flowers", help="dataset directory")

parser.add_argument('--save_dir', action="store", default="", help="saving directory for checkpoint", dest="save_directory")

parser.add_argument('--arch', action="store", default="densenet201", choices=['vgg16', 'densenet201'],help="you can only choose vgg16 or densenet201", dest="architecture")

arser.add_argument('--learning_rate', action="store", default="0.0007", type=float, help="Set Learning rate",dest="learning_rate")

parser.add_argument('--hidden_units', action="store", nargs=3, default=[1920, 710, 200], type=int, help="enter 3 integers between 25088 and 102 in decreasing order",dest="hidden_units")

parser.add_argument('--epochs', action="store", default=30, type=int, help="set epochs", dest="epochs")

parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

args = parser.parse_args()

arg_data_dir =  args.data_directory
arg_save_dir =  args.save_directory
arg_architecture =  args.architecture
arg_lr = args.learning_rate
arg_hidden_units = args.hidden_units
arg_epochs = args.epochs

if args.gpu and torch.cuda.is_available(): 
    arg_gpu = args.gpu

elif args.gpu:
    arg_gpu = False
    print('GPU is not available, will use CPU...')
    print()

else:
    arg_gpu = args.gpu

print()
print("Data directory: root/{}/ \nSave directory: root/{} \nArchitecture: {} ".format(arg_data_dir, arg_save_dir, arg_architecture))
print('Learning_rate: ', arg_lr)
print('Hidden units: ', arg_hidden_units)
print('Epochs: ', arg_epochs)
print('GPU: ', arg_gpu)
print()

if 102 <= arg_hidden_units[2] <= arg_hidden_units[1] <= arg_hidden_units[0] <= 1920:
    print("Hidden units are OK.") 
    print()
else:
    arg_hidden_units.extend([1920, 710, 200)
    for i in range(3):
        arg_hidden_units.pop(0)

    print("Hidden units are incompatible with the model. Default hidden units {} will be used".format(arg_hidden_units))
    print()


data_dir = arg_data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
                             
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
                             
if arg_architecture == 'vgg16':
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
                             
    model.classifier = nn.Sequential(nn.Linear(25088, arg_hidden_units[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(arg_hidden_units[0], arg_hidden_units[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(arg_hidden_units[1], arg_hidden_units[2]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),                                 
                                 nn.Linear(arg_hidden_units[2], 102),
                                 nn.LogSoftmax(dim=1)
    print('Model Classifier: ')
    print(model.classifier)
    print()
            

else :
    model = models.densenet201(pretrained=True)
                                     
    for param in model.parameters():
        param.requires_grad = False
                                     
    model.classifier = nn.Sequential(nn.Linear(1920, 710),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(710, 200),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(200, 102),
                                     nn.LogSoftmax(dim=1))
    
    print('Model Classifier: ')
    print(model.classifier)
    print()                         


criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.0007)

model.to(device);
                         
print()                                     
print('Training the model')
print('Do not turn off your computer')       
print()
                                     
epochs = 30

train_losses, valid_losses = [], []
for e in range(epochs):
    tot_train_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        tot_train_loss += loss.item()
        
    tot_valid_loss = 0
    valid_correct = 0  

    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            tot_valid_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_correct += torch.mean(equals.type(torch.FloatTensor)).item()

    train_loss = tot_train_loss / len(trainloader)
    valid_loss = tot_valid_loss / len(validloader)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss),
          "Valid Loss: {:.3f}.. ".format(valid_loss),
          "Valid Accuracy: {:.3f}".format(valid_correct / len(validloader)))
                                     
print()
print('Testing the model')
print('Do not turn off your computer')                                     
print()
                                     
epochs = 30

train_losses, test_losses = [], []
for e in range(epochs):
    tot_train_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        tot_train_loss += loss.item()
        
    tot_test_loss = 0
    test_correct = 0  

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            tot_test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_correct += torch.mean(equals.type(torch.FloatTensor)).item()

    train_loss = tot_train_loss / len(trainloader)
    test_loss = tot_test_loss / len(testloader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss),
          "Test Loss: {:.3f}.. ".format(test_loss),
          "Test Accuracy: {:.3f}".format(test_correct / len(testloader)))
                                     
print()
print('Saving the model')
print('Do not turn off your computer')
print()

                                     
if arg_save_dir:
    if not os.path.exists(arg_save_dir):
        os.mkdir(arg_save_dir)
        print("Directory " , arg_save_dir ,  " has been created for saving checkpoints")
    else:
        print("Directory " , arg_save_dir ,  " allready exists for saving checkpoints")
    save_dir = arg_save_dir + '/checkpoint.pth'
else:
    save_dir = 'checkpoint.pth'

print()                                


model.class_to_idx = [train_data,valid_data,test_data][0].class_to_idx
                                     
model.cpu()
checkpoint = {'input_size': 1920,
              'output_size': 102,
              'arch': 'densenet201',
              'learning_rate':0.0007,
              'batch_size': 64,
              'classifier' : model.classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')                                     

def loadCheckpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer, input_size, output_size, epoch
                                     
