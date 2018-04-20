import cv2
from os import listdir, path, makedirs

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn import linear_model


def feature_extraction(img):
    img_greyscale = rgb2gray(img)
    lbp = local_binary_pattern(img_greyscale, n_points, radius, 'ror')
    (hist, _, _) = plt.hist(lbp.ravel(), bins=255, range=(0, 256))
    #plt.show()
    return hist



# settings for LBP
radius = 1
n_points = 8 * radius

dirName = 'train'
classes = listdir(dirName)
classesDic = dict(zip(classes, range(len(classes))))

trainSamples = []
trainClasses = []
for clazz in classes:
    print(clazz)
    for image in listdir(dirName + '/' + clazz):
        print(image)
        completeFilePath = dirName + '/' + clazz + '/' + image
        img = cv2.imread(completeFilePath)
        sample = feature_extraction(img)
        trainSamples.append([sample])
        trainClasses.append([classesDic[clazz]])

# independently normalize each sample
trainSamples = normalize(np.array(trainSamples), axis=1)
trainClasses = np.array(trainClasses)

logreg = linear_model.LogisticRegressionCV(Cs=10, cv=5, dual=False, penalty='l2', n_jobs=-1)

logreg.fit(trainSamples, trainClasses)
