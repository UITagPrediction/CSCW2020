import sys
import tqdm
import numpy as np
sys.path.append('../../Data/')
import categorization

WORD2VEC = "../../Data/glove.6B.50d.txt"

# loading categorization
def loadCategory():
    category = categorization.categorization()
    return category

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