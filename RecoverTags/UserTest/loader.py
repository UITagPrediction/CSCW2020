import torch
import torch.utils.data
import torchvision.transforms 
from torchvision import datasets
import numpy as np
import random
from function import loadCategory
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000   

VECTOR_DIM = 50

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
def image_generator():
    DATA_PATH_TEST = './images/'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])
    test_loader = ImageFolderWithPaths(root=DATA_PATH_TEST, transform=transform)
    print(len(test_loader)," images are loaded.....")
    test_loader = torch.utils.data.DataLoader(dataset=test_loader,batch_size=32,shuffle=False)
    return test_loader

# Tag Dataloader
def tag_generator(paths, df, model, t):
    tags_input = []
    for path in paths:
        id = int(path.split('/')[-1].split('.')[0])
        tags = df.loc[df['id'] == id].readin.tolist()[0]
        try:
            tags = [x for x in tags if x not in loadCategory()[t]]
        except:
            pass
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
    test_loader = image_generator()
