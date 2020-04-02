# -*- coding: utf-8 -*-
"""
Created on Wed June 10 08:51:32 2019

@author: Sidong Feng
"""
import time
import os, copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import model
from loader import image_generator, tag_generator
from function import load_image_tags, loadGloveModel, loadCategory

parser = argparse.ArgumentParser(description='Recovering Missing Semantics')
parser.add_argument('--tag', type=str, default='blue',
                    help='tag to train (default: blue)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--weight', type=float, default=5e-3,
                    help='initial weight decay (default: 5e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--model', type=str, default='resnet',
                    help='model architecture (default: resnet)')
parser.add_argument('--mode', action='store_false',
                    help='finetune the partial model (default: true)')
parser.add_argument('--pretrain', action='store_false',
                    help='use pretrain network (default: true)')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

# data loader
img_tags = load_image_tags(args.tag)
w2v = loadGloveModel()
train_loader, valid_loader, test_loader = image_generator(args.tag)

# Hyperparameters
epochs = args.epochs
learning_rate = args.lr
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
model_name = args.model
feature_extract = args.mode
model_image, _ = model.initialize_model(model_name, feature_extract, use_pretrained=args.pretrain)
model_tag = model.CNN()
model = model.MyEnsemble(model_image, model_tag)
model = model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("-"*10)
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
print('-'*10)
optimizer = getattr(optim, args.optim)(params_to_update, lr=learning_rate, weight_decay=args.weight)
criterion = nn.CrossEntropyLoss()

""" Train model

        Example:
        >>> loss, accuracy = train_model()
        
"""
def train_model():
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for img_inputs, labels, paths in train_loader:
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)
        tag_inputs = tag_generator(paths, img_tags, w2v).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(img_inputs, tag_inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * img_inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

""" Valid model

        Example:
        >>> loss, accuracy = valid_model()
        
"""
def valid_model():
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for img_inputs, labels, paths in valid_loader:
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)
        tag_inputs = tag_generator(paths, img_tags, w2v).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(img_inputs, tag_inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * img_inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_corrects.double() / len(valid_loader.dataset)
    print('valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

""" Test model

        Example:
        >>> loss, accuracy = test_model()
        
"""
def test_model():
    model.load_state_dict(torch.load(args.tag+'_backup/checkpoint.pt'))
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for img_inputs, labels, paths in test_loader:
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)
        tag_inputs = tag_generator(paths, img_tags, w2v).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(img_inputs, tag_inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * img_inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

""" 
    
    Main function
        
"""
def main():
    # train and valid model
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        train_loss, train_acc = train_model()
        valid_loss, valid_acc = valid_model()
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)
        if valid_acc>best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    # save best accuracy on valid dataset
    print('Best val Acc: {:4f}'.format(best_acc))
    if not os.path.exists(args.tag+'_backup'):
        os.mkdir(args.tag+'_backup')
    torch.save(best_model_wts, args.tag+'_backup/checkpoint.pt')
    
    # visualise loss diagram and accuracy diagram
    plt.figure(1)
    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation', 'Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.figure(2)
    plt.plot(train_acc_history)
    plt.plot(valid_acc_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation','Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy: %')
    plt.show()

    # test model
    test_model()

if __name__ == "__main__":
    print(args)
    print('-'*20)
    start_time = time.time()
    main()
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))