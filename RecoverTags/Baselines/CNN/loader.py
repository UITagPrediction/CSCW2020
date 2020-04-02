import os
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import sys
sys.path.append('../../Data/')
from function import loadCategory
from function import loadGloveModel

VECTOR_DIM = 50
BATCH_SZ = 32

# load Metadata.csv
df = pd.read_csv("../../Data/Metadata.csv", encoding = "ISO-8859-15", header=None, low_memory=False)
# load category
category = loadCategory()
# load w2v
w2v = loadGloveModel()

# remove relevant tags and ui tags and bad tags
def remove_tags(string, t, category):
    try:
        string = string.split('   ')
        string = [x for x in string if x not in category['ui']]
        string = [x for x in string if x not in category[t]]
        string = [x.replace(' ','') for x in string]
        string = list(set(string))
        string = [x for x in string if len(x) > 1]
        string = [x for x in string if not x.isdigit()]
        string = [x.lower() for x in string]
        string.sort(key = len)
    except:
        string = []
    return string

def construct_loader(dict,tag):
    input = []
    target = [1]*len(dict['y'])+[0]*len(dict['n'])
    for id in dict['y']+dict['n']:
        try:
            clean_tags = remove_tags(df[df[0]==id].iloc[0][5], tag, category)
        except:
            clean_tags = []
        vecs = []
        for c_t in clean_tags:
            try:
                vec = w2v[c_t]
                vecs.append(vec)
            except:
                pass
        # completing to 1*VECTOR_DIM*VECTOR_DIM
        zeros = np.zeros((VECTOR_DIM,), dtype=float)
        for _ in range(VECTOR_DIM-len(vecs)):
            vecs.append(zeros)
        vecs = np.asarray(vecs)
        vecs = vecs.flatten()
        input.append(vecs.reshape(-1))
    # reform input as (BATCH_SZ*1*VECTOR_DIM*VECTOR_DIM)
    input = np.array(input).reshape(-1,1,VECTOR_DIM,VECTOR_DIM)
    input = torch.Tensor(input).float()
    target = torch.Tensor(target).long()
    return input, target


# Data Dataloader
def data_generator(tag, BATCH_SZ=BATCH_SZ):
    # load train, valid, test information
    DATA_PATH_TRAIN = '../../Data/'+tag+'/train/'
    DATA_PATH_VALID = '../../Data/'+tag+'/valid/'
    DATA_PATH_TEST = '../../Data/'+tag+'/test/'
    train_imgs = {'y':os.listdir(DATA_PATH_TRAIN+'y'), 'n':os.listdir(DATA_PATH_TRAIN+'n')}
    if '.DS_Store' in train_imgs['y']: 
        train_imgs['y'].remove('.DS_Store')
    if '.DS_Store' in train_imgs['n']: 
        train_imgs['n'].remove('.DS_Store')
    valid_imgs = {'y':os.listdir(DATA_PATH_VALID+'y'), 'n':os.listdir(DATA_PATH_VALID+'n')}
    if '.DS_Store' in valid_imgs['y']: 
        valid_imgs['y'].remove('.DS_Store')
    if '.DS_Store' in valid_imgs['n']: 
        valid_imgs['n'].remove('.DS_Store')
    test_imgs = {'y':os.listdir(DATA_PATH_TEST+'y'), 'n':os.listdir(DATA_PATH_TEST+'n')}
    if '.DS_Store' in test_imgs['y']: 
        test_imgs['y'].remove('.DS_Store')
    if '.DS_Store' in test_imgs['n']: 
        test_imgs['n'].remove('.DS_Store')
    for k in train_imgs.keys():
        train_imgs[k] = [int(x.split('.')[0].rsplit('_')[-1]) for x in train_imgs[k]]
    for k in valid_imgs.keys():
        valid_imgs[k] = [int(x.split('.')[0].rsplit('_')[-1]) for x in valid_imgs[k]]
    for k in test_imgs.keys():
        test_imgs[k] = [int(x.split('.')[0].rsplit('_')[-1]) for x in test_imgs[k]]
    # # for platform
    # train_id = [int(x.split('.')[0]) for x in train_imgs]
    # valid_id = [int(x.split('.')[0]) for x in valid_imgs]
    # test_id = [int(x.split('.')[0]) for x in test_imgs]
    
    # construct train, valid, test dataloader
    train_input,train_target = construct_loader(train_imgs,tag)
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(train_input,train_target),
        batch_size=BATCH_SZ, 
        shuffle=True,               
    )
    valid_input,valid_target = construct_loader(valid_imgs,tag)
    valid_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(valid_input,valid_target),
        batch_size=len(valid_input), 
        shuffle=True,               
    )
    test_input,test_target = construct_loader(test_imgs,tag)
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(test_input,test_target),
        batch_size=len(test_input), 
        shuffle=True,               
    )
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    """ generate data loader """
    train_loader, valid_loader, test_loader = data_generator('green')
    for input, target in valid_loader:
        print(input.shape)
        print(target.shape)
    