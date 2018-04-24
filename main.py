from os import listdir, path, makedirs

import numpy as np
from skimage.restoration import denoise_wavelet
from skimage import io, measure, feature;io.use_plugin('matplotlib')
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


def create_finger_print(img_path):
    print(img_path)
    img = io.imread(img_path)
    img = centeredCrop(img, 504, 504)
    deionised = denoise_wavelet(img, multichannel=True, )
    return img.astype('float') * (img.astype('float') - deionised.astype('float')), (img.astype('float') ** 2)


def feature_extract(img_path):
    print(img_path)
    img = io.imread(img_path)
    img = centeredCrop(img, 504, 504)
    deionised = denoise_wavelet(img, multichannel=True, )
    finger_print = img.astype('float') * (img.astype('float') - deionised.astype('float')) / (img.astype('float') ** 2)
    finger_print = np.nan_to_num(finger_print, copy=False)
    finger_print = rescale(finger_print, 0., 1.)
    features_moment = np.zeros((7*3))
    for i in range(3):
        m = measure.moments(finger_print[:, :, i], order=5)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mc = measure.moments_central(finger_print[:, :, i], cr, cc, order=5)
        mn = measure.moments_normalized(mc)
        mu = measure.moments_hu(mn)
        features_moment[i*7:i*7+7] = mu

    features_correlation = []
    for colors in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
        for delta_i in range(4):
            for delta_j in range(4):
                offset_image = ndimage.shift(finger_print[:, :, colors[1]], (delta_i, delta_j))
                shift, error, diffphase = feature.register_translation(finger_print[:, :, colors[0]], offset_image)
                features_correlation.append(diffphase)

    features = np.append(features_moment, np.array(features_correlation))

    features_covariance = np.zeros((3*2**2)**2+(3*3**2)**2)
    for k in [2, 3]:
        blocks = feature_extraction.image.extract_patches(finger_print, (k, k, 3), k)
        blocks = blocks.ravel()
        blocks = np.split(blocks, len(blocks)//(3*k**2))
        features_covariance[(k-2)*((3*(k-1)**2)**2):(k-2)*((3*(k-1)**2)**2)+((3*k**2)**2)] = np.cov(blocks, rowvar=False).ravel()

    features = np.append(features, features_covariance)

    correlation = []
    for color in range(3):
        error = np.mean(finger_print[:, :, color], axis=1)
        cor = np.zeros(len(error))
        for i in range(len(error)):
            cor[i] = np.correlate(error, np.roll(error, i))
        cor = np.mean(np.split(cor, 8), axis=1)

    features = np.append(features, cor)

    return features


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
    print("Creating Finger print class " + str(classesDic[clazz]))
    fileName = "SPN_FP_" + clazz + ".npy"
    if path.isfile(fileName):
        fingerPrintSamples = np.load(fileName)
    else:
        fingerPrintSamples = Parallel(n_jobs=num_cores)(
            delayed(create_finger_print)(i) for i in completeFilesPath[classesDic[clazz]])
        fingerPrintSamples = np.array(fingerPrintSamples)
        np.save(fileName, fingerPrintSamples)

    if 'trainSamples' not in locals():
        trainSamples = clazzSamples
    else:
        trainSamples = np.append(trainSamples, clazzSamples, axis=0)

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

# apply PCA
pca = decomposition.PCA(n_components=4)
trainSamples[:, 7+0*4:7+1*4] = pca.fit_transform(trainSamples[:, 7+0*4:7+0*4 +  96])
trainSamples = np.delete(trainSamples, range(7+1*4,  96+7+0*4), 1)
trainSamples[:, 7+1*4:7+2*4] = pca.fit_transform(trainSamples[:, 7+1*4:7+1*4 + 144])
trainSamples = np.delete(trainSamples, range(7+2*4, 144+7+1*4), 1)
trainSamples[:, 7+2*4:7+3*4] = pca.fit_transform(trainSamples[:, 7+2*4:7+2*4 + 729])
trainSamples = np.delete(trainSamples, range(7+3*4, 729+7+2*4), 1)
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
