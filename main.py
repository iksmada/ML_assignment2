import cv2
from os import listdir, path, makedirs

import numpy as np
from skimage.restoration import denoise_wavelet
from sklearn import linear_model, metrics, preprocessing, model_selection
from scipy.stats import moment

from joblib import Parallel, delayed
import multiprocessing


def feature_extraction(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    deionised = denoise_wavelet(img, multichannel=True)
    finger_print = img * (deionised - img) / img**2
    return [moment(finger_print[:, :, 0], moment=1),
            moment(finger_print[:, :, 0], moment=2),
            moment(finger_print[:, :, 0], moment=3),
            moment(finger_print[:, :, 1], moment=1),
            moment(finger_print[:, :, 1], moment=2),
            moment(finger_print[:, :, 1], moment=3),
            moment(finger_print[:, :, 2], moment=1),
            moment(finger_print[:, :, 2], moment=2),
            moment(finger_print[:, :, 2], moment=3)]


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
        if len(classPaths)>=50: break
        classPaths.append(dirName + '/' + clazz + '/' + image)
        trainClasses.append(classesDic[clazz])
    completeFilesPath.append(classPaths)

for clazz in classes:
    print("Preprocessing class " + str(classesDic[clazz]))
    fileName = "SPN_" + clazz + ".npy"
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

print(classesDic)
print(metrics.confusion_matrix(y_test, pred, [k for k in classesDic.values()]))
print(metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred, [k for k in classesDic.values()], [k for k in classesDic.keys()]))