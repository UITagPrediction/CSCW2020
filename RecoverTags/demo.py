import os
import sys
import torch
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from dataloader import Dribbble_Demo
from model import hybrid_model


parser = argparse.ArgumentParser(description='Recovering Missing Tags')

parser.add_argument('--demo_path', type=str, default="Dribbble_test/black",
                    help='predicted image path')
parser.add_argument('--checkpoint_path', default="real_weights/black_epoch_500_acc_99.pth", type=str,
                    help='checkpoint state_dict path')
parser.add_argument('--tag', default=None, type=str,
                    help='demo tag')
parser.add_argument('--glove_path', type=str, default="glove.6B.50d.txt",
                    help='glove file')
parser.add_argument('--result_path', type=str, default="tmp/",
                    help='save the results to the directory')
parser.add_argument('--cuda', default="True", type=str,
                    help='Use CUDA to train model')

args = parser.parse_args()

if __name__ == '__main__':
    
    net = hybrid_model("resnet", True, True)
    net.load_state_dict(torch.load(args.checkpoint_path))
    if eval(args.cuda):
        net = net.cuda()

    dataloader = Dribbble_Demo(args.demo_path, args.glove_path, target_size=224)
    test_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=len(dataloader))
    for visual, textual in test_loader:
        if eval(args.cuda):
            visual = visual.cuda()
            textual = textual.cuda()

        outputs = net(visual, textual)
        preds = outputs.round()
        preds = torch.reshape(preds, (-1,))

        for i in range(len(preds)):
            image_path = dataloader.images_path[i]
            pred = preds[i]
            f = open(os.path.join(args.result_path, image_path.split(".")[0]+'.txt'), 'a')
            f.write(args.tag if pred==1 else "")
            f.close()