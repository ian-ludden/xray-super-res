'''
Compute the RMSE for a subset of 2000 training image pairs.
'''
import numpy as np
import os
from keras import backend as K

def computeRMSE(pre, tru):
    return K.sqrt(K.mean(K.square(pre - tru), axis=-1))


if __name__ == "__main__":
    home = os.path.expanduser("~")

    start = 4000
    end = 5999

    nFiles = end - start + 1
    pre = np.zeros((nFiles, 128, 128, 1))
    tru = np.zeros((nFiles, 128, 128, 1))

    for index in range(start, end+1):
        tru_filename = '{0}/Documents/xray-super-res/xray_images/train_images_128x128/train_{1:05d}.png'.format(home, index)
        # TODO: Predict some training images to compute RMSE. These files don't exist yet.
        pre_filename = '{0}/Documents/xray-super-res/xray_images/train_images_out/train_{1:05d}.png'.format(home, index)
