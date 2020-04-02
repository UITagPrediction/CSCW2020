#######################################

###########    Function    ############
        
#######################################
import sys
import csv
import tqdm
import pandas as pd
import numpy as np
sys.path.append('../Data/')
import categorization

WORD2VEC = "../Data/glove.6B.50d.txt"
FILE = "../Data/Metadata.csv"

# loading categorization
def loadCategory():
    category = categorization.categorization()
    return category

# preprocess Metadata.csv
def load_image_tags(tag,f=FILE):
    print('-'*10)
    print("Preprocessing Image_tag.....")
    print('-'*10)
    
    # preprocess Metadata.csv
    df = pd.read_csv(f, encoding = "ISO-8859-15", header=None, low_memory=False)
    df = df[[0,5]]
    category = loadCategory()
    
    img_tags = {}
    for _, row in tqdm.tqdm(df.iterrows()):
        img, string = row[0], row[5]
        try:
            string = string.split('   ')
            # remove UI tags
            string = [x for x in string if x not in category['ui']]
            string = [x for x in string if x not in category[tag]]
            string = [x.replace(' ','') for x in string]
            string = list(set(string))
            string = [x for x in string if len(x) > 1]
            string = [x for x in string if not x.isdigit()]
            string = [x.lower() for x in string]
            string.sort(key = len)
        except:
            string = []
        img_tags[img] = string
    return img_tags

# load Word2Vec model
def loadGloveModel(gloveFile=WORD2VEC):
    print('-'*10)
    print("Loading Glove Model.....")
    print('-'*10)
    f = open(gloveFile,'r')
    model = {}
    for line in tqdm.tqdm(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    # print("Done.",len(model)," words loaded!")
    return model

if __name__ == "__main__":
    # loading dictionary of tags for each image, regardless the related tags
    load_image_tags('blue')
    # loading a dictionary for Glove
    loadGloveModel()