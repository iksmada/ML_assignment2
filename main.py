from os import listdir, path, makedirs

import numpy as np
from skimage.restoration import denoise_wavelet
from skimage import io, measure, feature, img_as_float;io.use_plugin('matplotlib')
from sklearn import linear_model, metrics, model_selection, feature_extraction, decomposition
from scipy import ndimage

from joblib import Parallel, delayed
import multiprocessing


def rescale(X, min=0., max=255., min_original=-1, max_original=-1):
    """
        X numpy array like matrix
        if min_original or  max_original are not explict given,
        this funcition tries to use X.min() and X.max() to fit values
        """
    if min_original == -1 or max_original == -1:
        min_original = X.min()
        max_original = X.max()

    return ((max - min) / (max_original - min_original)) * (X - min_original) + min


def centeredCrop(img, new_height, new_width):
    width = np.size(img, 1)
    height = np.size(img, 0)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    c_img = img[top:bottom, left:right, :]
    return c_img


def create_finger_print(img_path, isUp):
    print(img_path)
    img = io.imread(img_path)
    img = centeredCrop(img, 504, 504)
    if isUp:
        deionised = denoise_wavelet(img, multichannel=True, )
        return img_as_float(img) * (img_as_float(img) - deionised)#, (img_as_float(img) ** 2)
    else:
        return img_as_float(img) ** 2


def feature_extract(img_path):
    print(img_path)
    img = io.imread(img_path)
    img = centeredCrop(img, 504, 504)

    # for color in range(3):
    #     for i in range(len(trainFingerPrint)):
    #         cor[i+color*len(trainFingerPrint)] = np.correlate(img[:, :, color], trainFingerPrint[i][:, :, color])
    features_correlation = []
    for colors in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
        for i in range(len(trainFingerPrint)):
            shift, error, diffphase = feature.register_translation(img[:, :, colors[0]], trainFingerPrint[i][:, :, colors[1]])
            features_correlation.append(diffphase)

    return np.array(features_correlation)


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
        if len(classPaths) >= 50: break
        classPaths.append(dirName + '/' + clazz + '/' + image)
        trainClasses.append(classesDic[clazz])
    completeFilesPath.append(classPaths)

trainFingerPrint = []
for clazz in classes:
    print("Creating Finger print class " + str(classesDic[clazz]))
    fileName = "SPN2_FP_" + clazz + ".npy"
    if path.isfile(fileName):
        fingerPrintSamples = np.load(fileName)
    else:
        up = Parallel(n_jobs=num_cores)(
            delayed(create_finger_print)(i, True) for i in completeFilesPath[classesDic[clazz]])
        up = np.sum(up, 0)
        down = Parallel(n_jobs=num_cores)(
            delayed(create_finger_print)(i, False)for i in completeFilesPath[classesDic[clazz]])
        down = np.sum(down, 0)
        fingerPrintSamples = up / down
        fingerPrintSamples = np.nan_to_num(fingerPrintSamples, copy=False)
        np.save(fileName, fingerPrintSamples)

    trainFingerPrint.append(fingerPrintSamples)

for clazz in classes:
    print("Preprocessing class " + str(classesDic[clazz]))
    fileName = "SPN2_" + clazz + ".npy"
    if path.isfile(fileName):
        clazzSamples = np.load(fileName)
    else:
        clazzSamples = Parallel(n_jobs=num_cores)(
            delayed(feature_extract)(i) for i in completeFilesPath[classesDic[clazz]])
        clazzSamples = np.array(clazzSamples)
        np.save(fileName, clazzSamples)

    if 'trainSamples' not in locals():
        trainSamples = clazzSamples
    else:
        trainSamples = np.append(trainSamples, clazzSamples, axis=0)

trainClasses = np.array(trainClasses)


X_train, X_test, y_train, y_test = model_selection.train_test_split(trainSamples, trainClasses, train_size=0.8)

logreg = linear_model.LogisticRegressionCV(Cs=10, cv=5, dual=False, penalty='l2', n_jobs=-1)

logreg.fit(X_train, y_train)

pred = logreg.predict(X_test)

print(classesDic)
cm = metrics.confusion_matrix(y_test, pred, [k for k in classesDic.values()])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print(cm)
print(metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred, [k for k in classesDic.values()], [k for k in classesDic.keys()]))
