import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import sys

img_width, img_height = 28, 28


def create_model():
    model = Sequential()

    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(36, activation='softmax'))

    model.summary()

    return model


# model predict number if true otherwise character
def prediction(model, img, number):
    # img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, (img_width, img_height))
    model = create_model()
    model.load_weights('./weights.h5')
    arr = numpy.array(img).reshape((img_width, img_height, 3))
    arr = numpy.expand_dims(arr, axis=0)
    prediction = model.predict(arr)[0]
    bestclass = ''
    bestconf = -1
    if number:
        for n in range(10):
            if (prediction[n] > bestconf):
                bestclass = int(n)
                bestconf = prediction[n]
        return str(bestclass)
    else:
        for n in range(10, 36):
            if (prediction[n] > bestconf):
                bestclass = int(n)
                bestconf = prediction[n]
        return chr(bestclass + 55)
