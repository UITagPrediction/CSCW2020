import os
import sys
import torch
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random

from dataloader import Dribbble
from model import hybrid_model

random.seed(42)

parser = argparse.ArgumentParser(description='Recovering Missing Tags')

parser.add_argument('--tag', type=str, default=None,
                    help='train tag')
parser.add_argument('--checkpoint_save_path', default="real_weights", type=str,
                    help='checkpoint state_dict file')
parser.add_argument('--resume', default=None, type=str,
                    help='resume from checkpoint state_dict file')
parser.add_argument('--batch_size', default=32, type = int,
                    help='batch size of training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-3, type=float,
                    help='Weight decay')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use')
parser.add_argument('--backbone', type=str, default='resnet',
                    help='backbone architecture')
parser.add_argument('--load_resnet_pretrain', type=str, default="True",
                    help='Load resnet pretrain network')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--glove_path', type=str, default="glove.6B.50d.txt",
                    help='glove file')
parser.add_argument('--train_path', type=str, default="Dribbble_train/checkout",
                    help='training data path')
parser.add_argument('--test_path', type=str, default=None,
                    help='testing data path')
parser.add_argument('--feature_extract', type=str, default="True",
                    help='feature exract on backbone')
parser.add_argument('--cuda', default="True", type=str,
                    help='Use CUDA to train model')
parser.add_argument('--multigpu', default="1", type=str,
                    help='activate multi gpu')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.multigpu

if __name__ == '__main__':
    
    net = hybrid_model(args.backbone, eval(args.feature_extract), eval(args.load_resnet_pretrain))
    if args.resume:
        print("--------------------- Resume from {} ----------------------------".format(args.resume))
        net.load_state_dict(torch.load(args.resume))
    if eval(args.cuda):
        net = net.cuda()
        net = torch.nn.DataParallel(net,device_ids=list(args.multigpu.split(",")))

    dataloader = Dribbble(args.train_path, args.tag, args.glove_path, target_size=224)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    if args.test_path:
        dataloader = Dribbble(args.test_path, args.tag, args.glove_path, target_size=224)
        test_loader = torch.utils.data.DataLoader(
            dataloader,
            batch_size=len(dataloader),
            num_workers=32,
            pin_memory=True)

    params_to_update = net.parameters()
    if eval(args.feature_extract):
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    optimizer = getattr(optim, args.optim)(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    net.train()
    final_corr_value = 0
    for epoch in range(args.epochs):
        loss_value = 0
        corr_value = 0
        st = time.time()
        for i, (visual, textual, gt) in enumerate(train_loader):
            gt = gt.float()
            if eval(args.cuda):
                visual = visual.cuda()
                textual = textual.cuda()
                gt = gt.cuda()

            outputs = net(visual, textual)
            optimizer.zero_grad()
            loss = criterion(outputs, gt.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            preds = torch.round(torch.sigmoid(outputs.detach()))
            corr_value += (preds.reshape(-1) == gt).sum().float()
            # if i == 0:
            #     print((preds == gt))
            
        et = time.time()
        final_corr_value = corr_value.double()/len(train_loader.dataset)
        print('epoch {}:({}/{}) batch || training time {} || training loss {} || training accuracy {}'.format(epoch, epoch, args.epochs, et-st, loss_value, final_corr_value))

        if args.test_path and epoch % 50 == 0:
            st = time.time()
            test_batch = iter(test_loader)
            visual, textual, gt = next(test_batch)
            gt = gt.float()
            if eval(args.cuda):
                visual = visual.cuda()
                textual = textual.cuda()
                gt = gt.cuda()
            outputs = net(visual, textual)
            loss = criterion(outputs, gt.unsqueeze(1))
            loss_value = loss.item()
            preds = torch.round(torch.sigmoid(outputs.detach()))
            corr_value = (preds.reshape(-1) == gt).sum().float()/len(test_loader.dataset)
            et = time.time()
            print('epoch {}:({}/{}) batch || testing time {} || testing loss {} || testing accuracy {}'.format(epoch, epoch, args.epochs, et-st, loss_value, corr_value))
        

    torch.save(net.state_dict(), '{}/{}_epoch_{}_acc_{}.pth'.format(args.checkpoint_save_path, args.tag, args.epochs, int(final_corr_value*100)))