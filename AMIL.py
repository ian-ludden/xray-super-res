'''
Model for x-ray image super-resolutionself.
Created by Adithya Murali (adithya5) and Ian Ludden (iludden2) for
CS 446 Final Project, Fall 2018.

Note: Our code is heavily based on that of GitHub user MarkPrecursor.
https://github.com/MarkPrecursor/EEDS-keras

We also learned Keras partially from the following tutorial:
https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
'''

from keras.models import Sequential
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras.optimizers import adam
from keras.layers.merge import concatenate, add
import prepare_data as pd
from keras import backend as K

import cv2
import os
import numpy as np

# Flag for using only first 100 xray images.
IS_SMALL_TEST = False


EPOCHS = 100 # Can increase later...
BATCH_SIZE = 32 # Large batch size seems to cause out of memory errors

WORKING_DIR = "/home/iludden2/Documents/xray-super-res"

def Res_block():
    _input = Input(shape=(None, None, 64))

    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(conv)

    out = add(inputs=[_input, conv])
    out = Activation('relu')(out)

    model = Model(inputs=_input, outputs=out)

    return model


def model_AMIL():
    width = 64
    height = 64
    depth = 1
    model = Sequential()
    # inputShape = (height, width, depth)
    inputShape = (height, width, depth)
    chanDim = -1

    # MarkPrecursor version:
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=inputShape))
    model.add(Conv2DTranspose(filters=32, kernel_size=(14, 14), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(16384))
    model.add(Reshape((128, 128, 1)))
    return model

def model_AMIL_deep():
    width = 64
    height = 64
    depth = 1
    # model = Sequential()
    # inputShape = (height, width, depth)
    inputShape = (height, width, depth)
    chanDim = -1

    #MarkPrecursor EED:
    _input = Input(shape=(None, None, 1), name='input')

    Feature = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    Feature_out = Res_block()(Feature)

    # Upsampling
    Upsampling1 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Feature_out)
    Upsampling2 = Conv2DTranspose(filters=4, kernel_size=(14, 14), strides=(2, 2),
                                  padding='same', activation='relu')(Upsampling1)
    Upsampling3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Upsampling2)

    # Mulyi-scale Reconstruction
    Reslayer1 = Res_block()(Upsampling3)

    Reslayer2 = Res_block()(Reslayer1)

    # ***************//
    Multi_scale1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Reslayer2)

    Multi_scale2a = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)

    Multi_scale2b = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2b = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2b)

    Multi_scale2c = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2c = Conv2D(filters=16, kernel_size=(5, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2c)

    Multi_scale2d = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2d = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2d)

    Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d])

    out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2)
    model = Model(input=_input, output=out)
    return model


def AMIL_train(func_model):

    AMIL = func_model()


    # TODO: Consider changing optimizer and/or loss function.
    # Can implement custom RMSE loss function as shown here:
    # https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
    AMIL.compile(optimizer=adam(lr=0.0003), loss='mse')

    # TODO: Ensure name of training data file is correct.
    train_file = "./train_first100.h5" if IS_SMALL_TEST else "./train.h5"
    # data, label = pd.prepare_training_data(IS_SMALL_TEST)
    data, label = pd.read_training_data(train_file)

    # TODO: Consider different batch size, number of epochs.
    # print(data.shape)
    # print(label.shape)
    # label = np.reshape(label, (100, 16384))
    # TODO: Cross-validation here
    AMIL.fit(data, label, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # TODO: Rename weights file if optimizer or loss changes.
    AMIL.save_weights("AMIL_deep_model_adam_mse.h5")


def AMIL_predict(func_model):
    AMIL = func_model()
    AMIL.load_weights("AMIL_deep_model_adam_mse.h5")

    #TODO: Loop over all test images and generate high-res predictions.
    xray_dir = "xray_images_first100" if IS_SMALL_TEST else "xray_images"
    test_lr_dir = "{0}/{1}/test_images_64x64".format(WORKING_DIR, xray_dir)
    test_hr_dir = "{0}/{1}/test_images_128x128".format(WORKING_DIR, xray_dir)

    # Load test images in as data
    filenames = sorted(os.listdir(test_lr_dir))
    nFiles = filenames.__len__()
    X = np.zeros((nFiles, 64, 64, 1))

    for index in range(nFiles):
        # print(filenames[index])
        filename = "{0}/{1}".format(test_lr_dir, filenames[index])
        img = cv2.imread(filename)
        if img is not None:
            X[index, :, :, 0] = img[:, :, 0]

    pre = AMIL.predict(X, batch_size=50)
    pre[pre[:] > 255] = 255
    pre = pre.astype(np.uint8)

    # Write high-res test images (to be submitted)
    img = np.zeros((128, 128, 1))
    for index in range(nFiles):
        img[:, :, 0] = np.reshape(pre[index], (128,128))
        outfilename = "{0}/test_{1:05d}.png".format(test_hr_dir, index + 1)
        cv2.imwrite(outfilename, img)


'''
Plot AMIL model using built-in Keras visualization.
'''
def AMIL_visualize():
    from keras.utils import plot_model
    AMIL = model_AMIL_deep()
    plot_model(AMIL, show_shapes=True, to_file='AMIL_model_eed.png')


if __name__ == "__main__":
    func_model = model_AMIL_deep
    AMIL_train(func_model)
    AMIL_predict(func_model)


# if __name__ == "__main__":
#     AMIL_visualize()
