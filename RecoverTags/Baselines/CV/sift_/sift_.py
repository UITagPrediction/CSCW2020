import cv2
import numpy as np

def calcSiftFeature(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create(200) # max number of SIFT points is 200
	kp, des = sift.detectAndCompute(gray, None)
	return des
 
def calcFeatVec(features, centers):
	featVec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):
		fi = features[i]
		diffMat = np.tile(fi, (50, 1)) - centers
		sqSum = (diffMat**2).sum(axis=1)
		dist = sqSum**0.5
		sortedIndices = dist.argsort()
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1	
	return featVec.ravel()

def learnVocabulary(featureSet):
	wordCnt = 50
	# use k-means to cluster a bag of features
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, _, centers = cv2.kmeans(featureSet, wordCnt, None, criteria, 20, flags)

	return centers
