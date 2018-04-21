import cv2
from os import listdir, path, makedirs

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, preprocessing, model_selection

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
    classPaths = []
    print("Creating paths on class " + str(classesDic[clazz]))
    for image in listdir(dirName + '/' + clazz):
        # if len(classPaths)>10: break
        classPaths.append(dirName + '/' + clazz + '/' + image)
        trainClasses.append(classesDic[clazz])
    completeFilesPath.append(classPaths)

for clazz in classes:
    print("Preprocessing class " + str(classesDic[clazz]))
    fileName = "LBP_" + clazz + ".npy"
    if path.isfile(fileName):
        clazzSamples = np.load(fileName)
    else:
        clazzSamples = Parallel(n_jobs=num_cores)(
            delayed(feature_extraction)(i) for i in completeFilesPath[classesDic[clazz]])
        clazzSamples = np.array(clazzSamples)
        np.save(fileName, clazzSamples)

    if 'trainSamples' not in locals():
        trainSamples = clazzSamples
    else:
        trainSamples = np.append(trainSamples, clazzSamples, axis=0)

# independently normalize each sample
trainSamples = preprocessing.normalize(np.array(trainSamples), axis=1)
trainClasses = np.array(trainClasses)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trainSamples, trainClasses, train_size=0.8)


logreg = linear_model.LogisticRegressionCV(Cs=10, cv=5, dual=False, penalty='l2', n_jobs=-1)

logreg.fit(X_train, y_train)

pred = logreg.predict(X_test)

print(metrics.confusion_matrix(y_test, pred))
print(metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred, [k for k in classesDic.values()], [k for k in classesDic.keys()]))