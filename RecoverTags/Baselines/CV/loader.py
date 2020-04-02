import os
import pandas as pd
import numpy as np
from hist.hist import hog_feature,color_histogram_hsv
from sift_.sift_ import calcSiftFeature, calcFeatVec, learnVocabulary

DATA_PATH = None
input_features = 0

def hist_loader(ftype):
    from PIL import Image
    df_train = pd.DataFrame(columns=['label']+['feature'+str(x) for x in range(input_features)])
    df_test = pd.DataFrame(columns=['label']+['feature'+str(x) for x in range(input_features)])
    
    # Train dataset
    no = 0
    for label in ['n','y']:
        for stage in ['train','valid']:
            imgs = os.listdir(DATA_PATH+'/'+stage+'/'+label)
            try:
                imgs.remove('.DS_Store')
            except:
                pass
            for img in imgs:
                if ftype == 'hist':
                    im = Image.open(DATA_PATH+'/'+stage+'/'+label+'/'+img)
                    im = im.resize((224,224))
                    im.load()
                    im_data = np.asarray(im, dtype="float")
                    feature = hog_feature(im_data)
                elif ftype == 'hsv':
                    im = Image.open(DATA_PATH+'/'+stage+'/'+label+'/'+img)
                    im.load()
                    im_data = np.asarray(im, dtype="float")
                    feature = color_histogram_hsv(im_data)
                else:
                    pass
                df_train.loc[no] = [0 if label=='n' else 1]+feature.tolist()
                no+=1
    
    # Test dataset
    no = 0
    for label in ['n','y']:
        imgs = os.listdir(DATA_PATH+'/test/'+label)
        try:
            imgs.remove('.DS_Store')
        except:
            pass
        for img in imgs:
            if ftype == 'hist':
                im = Image.open(DATA_PATH+'/test/'+label+'/'+img)
                im = im.resize((224,224))
                im.load()
                im_data = np.asarray(im, dtype="float")
                feature = hog_feature(im_data)
            elif ftype == 'hsv':
                im = Image.open(DATA_PATH+'/test/'+label+'/'+img)
                im.load()
                im_data = np.asarray(im, dtype="float")
                feature = color_histogram_hsv(im_data)
            else:
                pass
            df_test.loc[no] = [0 if label=='n' else 1]+feature.tolist()
            no+=1
    
    return df_train, df_test

def sift_loader(): 
    import cv2
    df_train = pd.DataFrame(columns=['label']+['feature'+str(x) for x in range(input_features)])
    df_test = pd.DataFrame(columns=['label']+['feature'+str(x) for x in range(input_features)])

    no = 0
    tno = 0
    for label in ['n','y']:
        featureSet = np.float32([]).reshape(0,128)
        features = []
        # Train dataset
        for stage in ['train','valid']:
            imgs = os.listdir(DATA_PATH+'/'+stage+'/'+label)
            try:
                imgs.remove('.DS_Store')
            except:
                pass
            for img in imgs:
                im = cv2.imread(DATA_PATH+'/'+stage+'/'+label+'/'+img)
                feature = calcSiftFeature(im)
                try:
                    featureSet = np.append(featureSet, feature, axis=0)
                    features.append(feature)
                except:
                    pass
        
        # learn vocabulary
        centers = learnVocabulary(featureSet)

        for feature in features:
            featVec = calcFeatVec(feature, centers)
            df_train.loc[no] = [0 if label=='n' else 1]+featVec.tolist()
            no+=1

        # Test dataset
        for stage in ['test']:
            imgs = os.listdir(DATA_PATH+'/'+stage+'/'+label)
            try:
                imgs.remove('.DS_Store')
            except:
                pass
            for img in imgs:
                im = cv2.imread(DATA_PATH+'/'+stage+'/'+label+'/'+img)
                feature = calcSiftFeature(im)
                featVec = calcFeatVec(feature, centers)
                df_test.loc[tno] = [0 if label=='n' else 1]+featVec.tolist()
                tno+=1
    
    return df_train,df_test

def data_loader(tag,ftype):
    global DATA_PATH
    global input_features

    DATA_PATH = "../../Data/"+tag

    if ftype == 'hist':
        input_features = 7056
    elif ftype == 'hsv':
        input_features = 10
    else:
        input_features = 50

    if ftype == 'hist' or ftype == 'hsv':
        return hist_loader(ftype)
    else:
        return sift_loader()


if __name__ == "__main__":
    df_train, df_test = data_loader('music','hist')
    print(df_train, df_test)

    