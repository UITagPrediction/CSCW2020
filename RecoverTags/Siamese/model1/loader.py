import os
import torch
import torch.utils.data
import torchvision.transforms 
from torchvision import datasets
import random
from PIL import Image
import numpy as np

BATCH_SZ = 32

def product_of_lists(a,b):
    from itertools import product
    res = [i for i in set(map(frozenset, product(a, b))) if len(i) > 1]
    return res

class UIsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_y = os.listdir(self.root_dir+'y')
        self.img_n = os.listdir(self.root_dir+'n')
        try: 
            self.img_y.remove(".DS_Store")
        except:
            pass
        try: 
            self.img_n.remove(".DS_Store")
        except:
            pass
        self.product = {"y": product_of_lists(self.img_y,self.img_y)}
        self.product["n"] = random.choices(product_of_lists(self.img_n,self.img_y+self.img_n),k=len(self.product["y"]))

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # label 1: postive
        if index < len(self.product["y"]):
            pair = list(self.product["y"][index])
            img1 = Image.open(self.root_dir+'y/'+pair[0])
            img2 = Image.open(self.root_dir+'y/'+pair[1])
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                # from matplotlib import pyplot as plt
                # img1 = torchvision.transforms.ToPILImage()(img1)
                # img1.show()
            return img1,img2,torch.from_numpy(np.array([1],dtype=np.float32))
        else:
            pair = list(self.product["n"][index-len(self.product["y"])])
            try:
                img1 = Image.open(self.root_dir+'n/'+pair[0])
            except:
                img1 = Image.open(self.root_dir+'y/'+pair[0])
            try:
                img2 = Image.open(self.root_dir+'n/'+pair[1])
            except:
                img2 = Image.open(self.root_dir+'y/'+pair[1])
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return img1,img2,torch.from_numpy(np.array([0],dtype=np.float32))
    
    def __len__(self):
        return len(self.product["y"]) + len(self.product["n"])

# Image Dataloader
def image_generator(tag, BATCH_SZ=BATCH_SZ):
    DATA_PATH_TRAIN = '../Data/'+tag+'/train/'
    DATA_PATH_VALID = '../Data/'+tag+'/valid/'
    DATA_PATH_TEST = '../Data/'+tag+'/test/'

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    transform_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=UIsDataset(root_dir=DATA_PATH_TRAIN, transform=transform_train),
        batch_size=BATCH_SZ, 
        shuffle=True,               
    )

    input = UIsDataset(root_dir=DATA_PATH_VALID, transform=transform_valid)
    valid_loader = torch.utils.data.DataLoader(
        dataset=input,
        batch_size=len(input), 
        shuffle=True,               
    )

    input = UIsDataset(root_dir=DATA_PATH_TEST, transform=transform_valid)
    test_loader = torch.utils.data.DataLoader(
        dataset=input,
        batch_size=len(input), 
        shuffle=True,               
    )
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    """ generate image loader """
    train_loader, valid_loader, test_loader = image_generator('sport')
    print(train_loader.dataset[0][1])
    # display image
    import PIL
    print(valid_loader.dataset[0][0])
    img = torchvision.transforms.ToPILImage()(valid_loader.dataset[0][0]).convert('RGB')
    img.show() 
    # # print class_to_idx
    # print(train_loader.dataset.class_to_idx)
    # # print size
    # print(len(train_loader.dataset))
    