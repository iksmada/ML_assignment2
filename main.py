import cv2
from os import listdir, path, makedirs

import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


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
    for image in listdir(dirName + '/' + clazz):
        completeFilePath = dirName + '/' + clazz + '/' + image
        img = cv2.imread(completeFilePath)
        sample = feature_extraction(img)
        trainSamples.append([sample])
        trainClasses.append([classesDic[clazz]])

trainSamples = np.array(trainSamples)
trainClasses = np.array(trainClasses)