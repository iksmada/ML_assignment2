import cv2
from os import listdir, path, makedirs

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, preprocessing

from joblib import Parallel, delayed
import multiprocessing


def feature_extraction(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    img = rgb2gray(img)
    lbp = local_binary_pattern(img, n_points, radius, 'ror')
    (hist, _, _) = plt.hist(lbp.ravel(), bins=255, range=(0, 256))
    # plt.show()
    return hist



# settings for LBP
radius = 1
n_points = 8 * radius

dirName = 'train'
classes = listdir(dirName)
classesDic = dict(zip(classes, range(len(classes))))

completeFilesPath = []
trainClasses = []

num_cores = multiprocessing.cpu_count()

for clazz in classes:
    for image in listdir(dirName + '/' + clazz):
        if len(completeFilesPath)>=10*(classesDic[clazz]+1): break
        completeFilesPath.append(dirName + '/' + clazz + '/' + image)
        trainClasses.append(classesDic[clazz])


print(trainClasses)
if path.isfile("LBP.npy"):
    trainSamples = np.load("LBP.npy")
else:
    trainSamples = Parallel(n_jobs=num_cores)(
        delayed(feature_extraction)(i) for i in completeFilesPath)

    trainSamples = preprocessing.normalize(np.array(trainSamples), axis=1)
    np.save("LBP.npy", trainSamples)

# independently normalize each sample
trainClasses = np.array(trainClasses)

logreg = linear_model.LogisticRegressionCV(Cs=10, cv=5, dual=False, penalty='l2', n_jobs=-1)

logreg.fit(trainSamples, trainClasses)

pred = logreg.predict(trainSamples)

print(metrics.confusion_matrix(trainClasses, pred))


