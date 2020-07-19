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
    return image_arr


def predict_num(image_file, _model):
    image = Image.open("IMG_0922.jpg")
    return find_max(_model.predict(transform(image).reshape(1, 28, 28)))


def show_data(stop):
    for arr in x_train[:stop]:
        im = Image.fromarray(arr)
        im.show()


def find_max(arr):
    max_num = 0
    max_index = 0
    dct = {}
    for index, value in enumerate(arr[0]):
        dct[index] = value
        if value > max_num:
            max_num = value
            max_index = index
    return dct, max_index


def binarize(arr):
    mean = np.mean(arr)
    arr[arr > mean] = 255
    arr[arr < mean] = 0
    return arr


def fit_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', use_bias=True),
        Dense(10, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save("MNIST model1")


def load_compile(filename="MNIST model1"):
    model = load_model(filename)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


if __name__ == "__main__":
    print(predict_num("IMG_0922.jpg", load_compile()))
