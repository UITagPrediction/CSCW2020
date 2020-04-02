#######################################

##########    Data Loader    ##########
        
#######################################
import torch
import torch.utils.data
import torchvision.transforms 
from torchvision import datasets
import numpy as np
import random

VECTOR_DIM = 50
BATCH_SZ = 32

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

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
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderWithPaths(root=DATA_PATH_TRAIN, transform=transform_train),
        batch_size=BATCH_SZ, 
        shuffle=True,               
    )

    input = ImageFolderWithPaths(root=DATA_PATH_VALID, transform=transform_valid)
    valid_loader = torch.utils.data.DataLoader(
        dataset=input,
        batch_size=len(input), 
        shuffle=True,               
    )

    input = ImageFolderWithPaths(root=DATA_PATH_TEST, transform=transform_valid)
    test_loader = torch.utils.data.DataLoader(
        dataset=input,
        batch_size=len(input), 
        shuffle=True,               
    )
    return train_loader, valid_loader, test_loader


# Tag Dataloader
def tag_generator(paths, img_tags, model):
    tags_input = []
    for path in paths:
        id = path.split('/')[-1].split('_')[-1].split('.')[0]
        try:
            tags = img_tags[int(id)]
        except:
            tags = []
        vecs = []
        for t in tags:
            try:
                vec = model[t]
                vecs.append(vec)
            except:
                pass
        # completing to 1*50*50
        zeros = np.zeros((VECTOR_DIM,), dtype=float)
        for _ in range(VECTOR_DIM-len(vecs)):
            vecs.append(zeros)
        vecs = np.asarray(vecs)
        vecs = vecs.flatten()
        tags_input.append(vecs.reshape(-1))
    tags_input = np.array(tags_input).reshape(-1,1,VECTOR_DIM,VECTOR_DIM)
    tags_input = torch.Tensor(tags_input).float()
    return tags_input

if __name__ == "__main__":
    """ generate image loader """
    train_loader, valid_loader, test_loader = image_generator('yellow')
    