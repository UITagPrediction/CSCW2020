import sys
import csv
import tqdm
import pandas as pd
import numpy as np
sys.path.append('../Data/')
import categorization

WORD2VEC = "../Data/glove.6B.50d.txt"

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

# loading categorization
def loadCategory():
    category = categorization.categorization()
    return category

# preprocess Meta.csv
def preprocess():
    print('-'*10)
    print("Preprocessing Meta.csv.....")
    print('-'*10)
    # read Meta.csv
    df = pd.read_csv('Meta.csv')
    category = loadCategory()
    df['normalise'] = ""
    df['readin'] = ""
    # add normalise, readin column
    for i, row in df.iterrows():
        try:
            # origin_tags
            origin_tags = row.origin.lower().split('+')
            # normalised_tags
            normalised_tags = []
            for tag in origin_tags:
                check = False
                for k, v in category.items():
                    if tag in v:
                        check = True
                        normalised_tags.append(k)
                if not check:
                    normalised_tags.append(tag)
            normalised_tags = list(set(normalised_tags))
            normalised_tags = ['list' if x == 'list_' else x for x in normalised_tags]
            # readin_tags
            readin_tags = [x for x in origin_tags if x not in category['ui']]
            readin_tags = [x.replace(' ','') for x in readin_tags]
            readin_tags = list(set(readin_tags))
            readin_tags = [x for x in readin_tags if len(x) > 1]
            readin_tags = [x for x in readin_tags if not x.isdigit()]
            readin_tags = [x.lower() for x in readin_tags]
            readin_tags.sort(key = len)
        except:
            normalised_tags = []
            readin_tags = []
        df.at[i,'normalise'] = normalised_tags
        df.at[i,'readin'] = readin_tags
    return df

# load Word2Vec model
def loadGloveModel(gloveFile=WORD2VEC):
    print('-'*10)
    print("Loading Glove Model.....")
    print('-'*10)
    f = open(gloveFile,'r',encoding='UTF-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    # print("Done.",len(model)," words loaded!")
    return model

# get all id by tag
def get_pre_tag_id(df,t):
    R = []
    for _, row in df.iterrows():
        if t in row.readin:
            R.append(row.id)
    return R

def write_json(df,return_result):
    df = df.drop(['readin'],axis=1)
    df['new'] = ""
    for i, row in df.iterrows():
        tags_ratio = {}
        for ele in return_result[row.id]:
            tags_ratio[ele[0]] = ele[1]
        # R includes all the normalised tag with a ratio by model or 1 for unmodel
        R = {}
        for t in row.normalise:
            R[t] = 1
        for t in tags_ratio.keys():
            R[t] = tags_ratio[t]
        # new tag
        new = list(set(tags_ratio.keys()).difference(set(row.origin.lower().split('+'))))
        df.at[i,'origin'] = row.origin.lower().split('+')
        df.at[i,'normalise'] = R
        df.at[i,'new'] = new

    fo = open("result.json","w",encoding='utf-8')
    fo.write(df.to_json(orient='records', force_ascii=False))
    fo.close()

if __name__ == "__main__":
    # # loading Meta.csv
    df = preprocess()

    # # loading a dictionary for Glove
    # loadGloveModel()

    write_json(df,{6441217:[('red',0.57)]})
    # print(a)
    # a = update_csv([3919542],"red",a)
    # write_csv(a)
    None