# -*- coding: utf-8 -*-
"""
Created on Wed June 10 08:51:32 2019

@author: Sidong Feng
"""
import os
import cv2
import tqdm
import random
import shutil 
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from autoaugment import ImageNetPolicy
from categorization import categorization,type__

FROM = './All_images/'

parser = argparse.ArgumentParser(description='Generating Dataset for Image and Tags')
parser.add_argument('--auto', action='store_true',
                    help='Image Auto Augmentation (default: false)')
parser.add_argument('--noise', action='store_true',
                    help='Image Noise Augmentation (default: false)')
parser.add_argument('--tag', type=str, default='blue',
                    help='Tag to generate dataset (default: blue)')
parser.add_argument('--scale', type=int, default=2,
                    help='Augmentation scale (default: 2)')
parser.add_argument('--resize', action='store_false',
                    help='Image resize (default: true)')
parser.add_argument('--resize_rate', type=int, default=224,
                    help='Image resize to 224*224 (default: 224)')
args = parser.parse_args()

def generate():
    """ Generate dataset including train, valid, test.

        Example:
        >>> generate()
        
    """
    # read vocabulary table and Metadata
    category = categorization()
    type_ = type__()
    df = pd.read_csv("Metadata.csv", encoding = "ISO-8859-15", header=None, low_memory=False)
    # check validation on tag
    if args.tag not in category.keys():
        print("Error: Invalid tag!")
        return
    # initial folder
    if os.path.exists(args.tag):
        shutil.rmtree(args.tag)
    os.mkdir(args.tag)
    global DATA_PATH_TRAIN, DATA_PATH_VALID, DATA_PATH_TEST
    DATA_PATH_TRAIN = args.tag+'/train/'
    DATA_PATH_VALID = args.tag+'/valid/'
    DATA_PATH_TEST = args.tag+'/test/'
    # Same-category tags as negative data
    negative = type_[[k for k, v in type_.items() if args.tag in v][0]]
    negative.remove(args.tag)
    print('#'*20)
    print("Start Dataset Generating ...")
    print('#'*20)
    error = 0
    imgs_list = {"y":[],"n":[]}
    for _, row in df.iterrows():
        try:
            tags = row[5].strip().split('   ')
        except:
            continue
        # Filter out non-UI image
        if len([value for value in tags if value in category['ui']])==0:
            continue
        # Positive data
        if len([value for value in tags if value in category[args.tag]])>0:
            imgs_list['y'].append(str(row[0]))
        # Negative data
        elif len([x for x in tags for y in negative if x in category[y]])>0:
            imgs_list['n'].append(str(row[0]))
        else:
            continue
    # Balance positive and negative data
    imgs_list['n'] = random.choices(imgs_list['n'],k=len(imgs_list['y']))
    imgs_list['all'] = imgs_list['y'] + imgs_list['n']
    random.shuffle(imgs_list['all'])
    random.shuffle(imgs_list['all'])
    # Random split dataset into train(0.8)/valid(0.1)/test(0.1)
    ratio_train = 0.8
    ratio_validation = 0.1
    p1 = int(ratio_train * len(imgs_list['all']))
    p2 = int(ratio_validation * len(imgs_list['all']))
    train = imgs_list['all'][:p1]
    valid = imgs_list['all'][p1:p1+p2]
    test = imgs_list['all'][p1+p2:]
    print('Train:', len(train))
    print('Valid:', len(valid))
    print('Test:', len(test))
    paths = {DATA_PATH_TRAIN:train, DATA_PATH_VALID:valid, DATA_PATH_TEST:test}
    # save images
    for k,v in paths.items():
        os.mkdir(k)
        os.mkdir(k+'y')
        os.mkdir(k+'n')
        for p in v:
            if os.path.exists(FROM+p+'.jpg'):
                p = p+'.jpg'
            elif os.path.exists(FROM+p+'.png'):
                p = p+'.png'
            else:
                continue
            try:
                img = Image.open(FROM+p)
                img = img.convert("RGB")
                # Optional: Resize
                if args.resize:
                    img = img.resize((args.resize_rate,args.resize_rate))
                if p.split('.')[0] in imgs_list['y']:
                    img.save(k+'y/'+p,quality=80)
                else:
                    img.save(k+'n/'+p,quality=80)
            except:
                error+=1
                continue   
    print("Occur",error,"error images")
    return imgs_list,train,valid,test

# Auto Augmentation on Dataset
def data_auto_augmentation(scale=2):
    """ Auto Augmentation on train dataset.

        Example:
        >>> data_auto_augmentation(scale=2)
        
    """
    print('#'*20)
    print("Start Image augmentation...")
    print('#'*20)
    path_y = DATA_PATH_TRAIN+'y/'
    save_dir_y = DATA_PATH_TRAIN+'y/'
    path_n = DATA_PATH_TRAIN+'n/'
    save_dir_n = DATA_PATH_TRAIN+'n/'
    # add augmentation on dataset
    j = 0
    for i in tqdm.tqdm(os.listdir(path_y)):
        try:
            for _ in range(scale):
                img = PIL.Image.open(path_y + i)
                policy = ImageNetPolicy()
                img1 = policy(img)
                img1.save(save_dir_y + '{}.jpg'.format(j))
                j += 1
        except:
            pass 
    for i in tqdm.tqdm(os.listdir(path_n)):
        try:
            for _ in range(scale):
                img = PIL.Image.open(path_n + i)
                policy = ImageNetPolicy()
                img1 = policy(img)
                img1.save(save_dir_n + '{}.jpg'.format(j)) 
                j += 1
        except:
            pass 
    print("Done.....")

def add_gasuss_noise(image, mean=0, var=0.0001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def data_noise_augmentation():
    """ Add noise on train dataset.

        Example:
        >>> data_noise_augmentation()
        
    """
    print('#'*20)
    print("Start Image augmentation...")
    print('#'*20)
    path_y = DATA_PATH_TRAIN+'y/'
    save_dir_y = DATA_PATH_TRAIN+'y/'
    path_n = DATA_PATH_TRAIN+'n/'
    save_dir_n = DATA_PATH_TRAIN+'n/'
    # add noise on dataset
    j = 0
    for i in tqdm.tqdm(os.listdir(path_y)):
        try:
            img = cv2.imread(path_y + i)
            img_out = add_gasuss_noise(img)
            cv2.imwrite(save_dir_y + 'noise_{}.jpg'.format(j), img_out)
            j += 1
        except:
            pass

    for i in tqdm.tqdm(os.listdir(path_n)):
        try:
            img = cv2.imread(path_n + i)
            img_out = add_gasuss_noise(img)
            cv2.imwrite(save_dir_n + 'noise_{}.jpg'.format(j), img_out)
            j += 1
        except:
            pass 
    print("Done.....")

# main function
def main():
    """ generate dataset """
    generate()
    """ add noise """
    if args.noise:
        data_noise_augmentation()
    """ add augmentation """
    if args.auto:
        data_auto_augmentation(args.scale)

if __name__ == "__main__":
    print('-'*10)
    print(args)
    print('-'*10)
    # main()
    keys = ['mobile', 'website', 'chart', 'grid', 'form', 'list', 'dashboard', 'profile', 'signup', 'checkout', 'landing','weather', 'sport', 'game', 'finance', 'travel', 'food', 'ecommerce', 'music', 'pink', 'black', 'white', 'green', 'blue', 'red', 'yellow']
    r = [13233, 11065, 342, 399, 456, 301, 1785, 677, 1152, 699, 2161, 310, 520, 572, 890, 994, 925, 1772, 1216, 321, 1268, 719, 447, 1021, 376, 521]
    import matplotlib.pyplot as plt
    plt.bar(keys,r,align='center') # A bar chart
    plt.ylabel('# design images',fontsize=12)
    plt.xticks(rotation=-60, fontsize=12)
    plt.show()
    print(min(r))
    print(max(r))