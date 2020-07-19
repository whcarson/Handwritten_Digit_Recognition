import sys
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.datasets import mnist
import sklearn.datasets
import tensorflow.compat.v2 as tf
from PIL import ImageOps, Image, ImageEnhance
import numpy as np

tf.enable_v2_behavior()

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")


def transform(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = ImageEnhance.Contrast(image).enhance(2)
    image_arr = np.array(image)
    image_arr = binarize(image_arr)
    image = Image.fromarray(image_arr)
    image.show()
    return image_arr


def predict_num(image, _model):
    return find_max(_model.predict(transform(image).reshape(1, 28, 28)))


def show_data(stop):
    for arr in x_train[:stop]:
        im = Image.fromarray(arr)
        im.show()


def find_max(arr):
    max_num = 0
    max_index = 0
    for index, value in enumerate(arr[0]):
        print(index, value)
        if value > max_num:
            max_num = value
            max_index = index
    return max_index


def binarize(arr):
    mean = np.mean(arr)
    arr[arr > mean] = 255
    arr[arr < mean] = 0
    return arr


if __name__ == "__main__":
    im = Image.open("IMG_0922.jpg")
    # print(transform(im))
    transform(im)
    # model = Sequential([
    #     Flatten(input_shape=(28, 28)),
    #     Dense(128, activation='relu', use_bias=True),
    #     Dense(10, activation='softmax')
    # ])

    model = load_model("MNIST model1")

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )

    # model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    # model.save("MNIST model1")
    print(predict_num(im, model))
    # show_data(10)
