import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms 

from categorization import categorization,type__

class Dribbble(data.Dataset):
    def __init__(self, train_folder, tag, GlovePath, target_size=224):
        self.img_folder = os.path.join(train_folder, 'Images')
        self.tag_folder = os.path.join(train_folder, 'Tags')
        self.tag = tag
        self.target_size = target_size
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)
        self.category = categorization()
        self.glove = self.load_Glove_Model(GlovePath)

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)

    def pull_item(self, index):
        visual = self.load_visual(index)
        textual, gt = self.load_textual_and_gt(index, self.tag, self.glove)
        return visual, textual, gt

    def get_imagename(self, index):
        return self.images_path[index]

    def load_visual(self, index):
        '''
        According index to load visual semantics
        :param index
        :return: visual semantics
        '''
        imagename = self.images_path[index]
        image_path = os.path.join(self.img_folder, imagename)
        
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(self.target_size),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        image = data_transform(Image.open(image_path))

        return image

    def load_textual_and_gt(self, index, tag, glove):
        '''
        According index to load textual semantics
        :param index
        :return: textual semantics
        '''
        tag_path = os.path.join(self.tag_folder, "%s.txt" % os.path.splitext(self.images_path[index])[0])
        lines = open(tag_path, encoding='utf-8').readlines()
        if len(lines)>0:
            words = lines[0].strip().encode('utf-8').decode('utf-8-sig').split(',')
        else:
            words = []
        words = [x.strip() for x in words]
        subtracted_words = []
        gt = False
        for word in words:
            word = word.lower()
            if word not in self.category[tag]:
                subtracted_words.append(word)
            else:
                gt = True
        subtracted_words = [x.lower() for x in subtracted_words]
        textual_map = np.array([glove[word] for word in subtracted_words if word in glove.keys()])
        zeros = np.zeros(((50-len(textual_map), 50)), dtype=float)
        if len(textual_map) > 0:
            padding_textual_map = np.concatenate((textual_map, zeros))
        else:
            padding_textual_map = zeros
        padding_textual_map = padding_textual_map.reshape(1, 50, 50)
        padding_textual_map = torch.from_numpy(padding_textual_map).float()
        return padding_textual_map, 1 if gt else 0

    def load_Glove_Model(self, glovePath):
        f = open(glovePath,'r')
        glovemodel = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            glovemodel[word] = embedding
        return glovemodel

class Dribbble_Demo(data.Dataset):
    def __init__(self, demo_folder, GlovePath, target_size=224):
        self.img_folder = os.path.join(demo_folder, 'Images')
        self.tag_folder = os.path.join(demo_folder, 'Tags')
        self.target_size = target_size
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)
        self.glove = self.load_Glove_Model(GlovePath)

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)
    
    def pull_item(self, index):
        visual = self.load_visual(index)
        textual = self.load_textual(index, self.glove)
        return visual, textual

    def load_visual(self, index):
        '''
        According index to load visual semantics
        :param index
        :return: visual semantics
        '''
        imagename = self.images_path[index]
        image_path = os.path.join(self.img_folder, imagename)
        
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(self.target_size),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        image = data_transform(Image.open(image_path))

        return image

    def load_textual(self, index, glove):
        '''
        According index to load textual semantics
        :param index
        :return: textual semantics
        '''
        tag_path = os.path.join(self.tag_folder, "%s.txt" % os.path.splitext(self.images_path[index])[0])
        lines = open(tag_path, encoding='utf-8').readlines()
        words = lines[0].strip().encode('utf-8').decode('utf-8-sig').split(',')
        words = [x.lower().strip() for x in words]
        textual_map = np.array([glove[word] for word in words if word in glove.keys()])
        zeros = np.zeros(((50-len(textual_map), 50)), dtype=float)
        padding_textual_map = np.concatenate((textual_map, zeros))
        padding_textual_map = padding_textual_map.reshape(1, 50, 50)
        padding_textual_map = torch.from_numpy(padding_textual_map).float()
        return padding_textual_map

    def load_Glove_Model(self, glovePath):
        f = open(glovePath,'r')
        glovemodel = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            glovemodel[word] = embedding
        return glovemodel

if __name__ == '__main__':
    dataloader = Dribbble("./Dribbble_train","white", "glove.6B.50d.txt")
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    for index, (visual, textual, gt) in enumerate(train_loader):
        # print(visual)
        # print(textual)
        print(gt)