import os, copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from model import SiameseNetwork, ContrastiveLoss
from loader import image_generator
# from earlystop import EarlyStopping

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Recovering Missing Semantics')
parser.add_argument('--tag', type=str, default='blue',
                    help='model to train (default: blue)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--weight', type=float, default=5e-3,
                    help='initial weight decay (default: 5e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--patience', type=int, default=5,
                    help='number of patience to early stop (default: 5)')
parser.add_argument('--mode', action='store_false',
                    help='finetune the partial model (default: true)')
# parser.add_argument('--early', action='store_false',
#                     help='use Early Stop (default: true)')
parser.add_argument('--lr_decay', action='store_true',
                    help='lr decay (default: false)')
parser.add_argument('--pretrain', action='store_false',
                    help='use pretrain network (default: true)')

args = parser.parse_args()
# args = parser.parse_args(args=[])
torch.manual_seed(args.seed)
print('-'*10)

# data loader
train_loader, valid_loader, test_loader = image_generator(args.tag)

# Hyperparameters
epochs = args.epochs
learning_rate = args.lr
lr_decay_rate = 0.1
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
model_name = "resnet"
# feature_extract = args.mode
model = SiameseNetwork()
model = model.to(device)
# print(model)

# load pretrained model
# if args.backup:
#     if os.path.exists('backup/checkpoint.pt'):
#         model.load_state_dict(torch.load('backup/checkpoint.pt'))

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
print("Params to learn:")
params_to_update = model.parameters()
for name,param in model.named_parameters():
    if param.requires_grad == True:
        print("\t",name)
print('-'*10)
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)
# print('-'*10)

optimizer = getattr(optim, args.optim)(params_to_update, lr=learning_rate)
criterion = ContrastiveLoss()

######################    
# train the model #
######################
def train_model():
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            # _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * img1.size(0)
        # running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 0
    print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

######################    
# valid the model #
######################
def valid_model():
    model.eval() # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
        running_loss += loss.item() * img1.size(0)
        # running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = 0
    print('valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

# ######################    
# # test the model #
# ######################
# def test_model():
#     model.load_state_dict(torch.load('backup/checkpoint.pt'))
#     model.eval()
#     running_loss = 0.0
#     running_corrects = 0

#     for img_inputs, labels, paths in test_loader:
#         img_inputs = img_inputs.to(device)
#         labels = labels.to(device)
#         tag_inputs = tag_generator(paths, img_tags, w2v).to(device)

#         optimizer.zero_grad()
#         with torch.set_grad_enabled(False):
#             outputs = model(img_inputs, tag_inputs)
#             loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)
#         running_loss += loss.item() * img_inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data)
    
#     epoch_loss = running_loss / len(test_loader.dataset)
#     epoch_acc = running_corrects.double() / len(test_loader.dataset)
#     print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

if __name__ == "__main__":
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

        if args.lr_decay and epoch%50==0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate
            print(param_group['lr'])
    
    print('Best val Acc: {:4f}'.format(best_acc))
    if not os.path.exists('backup'):
        os.mkdir('backup')
    torch.save(best_model_wts, 'backup/checkpoint.pt')
    
    # visualise loss diagram and accuracy diagram
    plt.figure(1)
    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    # plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation', 'Early Stopping Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.figure(2)
    plt.plot(train_acc_history)
    plt.plot(valid_acc_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    # plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation','Early Stopping Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy: %')
    plt.show()

    # test_model()



    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))