
import numpy
import os
# import tensorflow.keras as keras
import csv
from tensorflow import keras
import cv2 as cv2
import re
from PIL import Image
from skimage import io
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import choice
import easypyplot

import tensorflow as tf


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# import data
trainPath = input('Enter path of train photos (160x160) (include final slash)')
testPath = input('Enter path of test photos')
trainLabelPath = input('Enter path of csv file that has sequential labels for every train photo')
testLabelPath = input('Enter path of csv file that has sequential labels for every test photo')
numclasses = input('Enter number of different people in files')
filelist = os.listdir(trainPath)

model = keras.models.load_model('facenet_keras.h5')

trainX = []
anno_train = []

testX = []
anno_test = []

filelist = sorted_aphanumeric(filelist)
with open(trainLabelPath) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        anno_train.append(row[0])

# anno_train = keras.utils.to_categorical(anno_train, 39)
for fname in filelist:
    img = io.imread(trainPath + fname, as_gray=False)
    img = img.reshape([160, 160, 3])
    trainX.append(img)
img_train = numpy.array(trainX)
# put data in arrays

#evaluate

testlist = sorted_aphanumeric(os.listdir(testPath))
for test in testlist:
    img = io.imread(testPath + test, as_gray=False)
    # img = img.reshape([160, 160, 3])
    testX.append(img)
img_test = numpy.array(testX)
with open(testLabelPath) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        anno_test.append(row[0])
anno_test = numpy.array(anno_test)

# convert to embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = numpy.asarray(newTrainX)
# convert testx to embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = numpy.asarray(newTestX)

print('Dataset: train=%d, test=%d' % (newTrainX.shape[0], newTestX.shape[0]))

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
testX = in_encoder.transform(newTestX)
out_encoder = LabelEncoder()
out_encoder.fit(anno_train)
trainy = out_encoder.transform(anno_train)
testy = out_encoder.transform(anno_test)
model = SVC(kernel='linear', probability=True)
model.fit(numpy.array(trainX), numpy.array(trainy))

# model.save('C:/Users/Ben/Downloads/FacialRecog_2.0.h5')

# scores = model.predict_classes(img_test, verbose=1)
# print(scores)
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
#yhat_test_prob = model.predict_proba(testX[0])
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])

random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun

title = '%s (%.3f)' % (predict_names[0], class_probability)



