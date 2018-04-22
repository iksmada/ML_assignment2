from os import listdir, path, makedirs

import numpy as np
from skimage.restoration import denoise_wavelet
from skimage import io, measure, feature;

io.use_plugin('matplotlib')
from sklearn import linear_model, metrics, model_selection
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


def feature_extraction(img_path):
    print(img_path)
    img = io.imread(img_path)
    img = centeredCrop(img, 500, 500)
    deionised = denoise_wavelet(img, multichannel=True, )
    finger_print = img.astype('float') * (img.astype('float') - deionised.astype('float')) / (img.astype('float') ** 2)
    finger_print = np.nan_to_num(finger_print, copy=False)
    # finger_print = rescale(finger_print, 0., 1.)
    # features = np.zeros((7*3))
    # for i in range(3):
    #     m = measure.moments(finger_print[:, :, i], order=5)
    #     cr = m[0, 1] / m[0, 0]
    #     cc = m[1, 0] / m[0, 0]
    #     mc = measure.moments_central(finger_print[:, :, i], cr, cc, order=5)
    #     mn = measure.moments_normalized(mc)
    #     mu = measure.moments_hu(mn)
    #     features[i*7:i*7+7] = mu

    features = []
    for colors in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
        for delta_i in range(4):
            for delta_j in range(4):
                offset_image = ndimage.shift(finger_print[:, :, colors[1]], (delta_i, delta_j))
                shift, error, diffphase = feature.register_translation(finger_print[:, :, colors[0]], offset_image)
                features.append(diffphase)

    return np.array(features)


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
        if len(classPaths) >= 20: break
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
trainClasses = np.array(trainClasses)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trainSamples, trainClasses, train_size=0.8)

logreg = linear_model.LogisticRegressionCV(Cs=10, cv=5, dual=False, penalty='l2', n_jobs=-1)

logreg.fit(X_train, y_train)

pred = logreg.predict(X_test)

print(classesDic)
print(metrics.confusion_matrix(y_test, pred, [k for k in classesDic.values()]))
print(metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred, [k for k in classesDic.values()], [k for k in classesDic.keys()]))
