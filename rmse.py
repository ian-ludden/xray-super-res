'''
Compute the RMSE for a subset of 2000 training image pairs.
'''

import cv2
import numpy as np
import os

def computeRMSE(pre, tru):
    return np.sqrt(np.mean(np.square(pre - tru), axis=1))


if __name__ == "__main__":
    home = os.path.expanduser("~")

    starts = [4000, 8000, 10000, 12000, 14000, 16000, 18000]
    for start in starts:
        end = start + 2000 - 1

        nFiles = end - start + 1
        pre = np.zeros((nFiles, 16384))
        tru = np.zeros((nFiles, 16384))

        for index in range(start, end+1):
            tru_filename = '{0}/Documents/CS446_ML_Final_Project/xray-super-res/xray_images/train_images_128x128/train_{1:05d}.png'.format(home, index)
            imTru = cv2.imread(tru_filename)[:, :, 0]
            tru[index-start, :] = np.reshape(imTru, (16384,))

            # TODO: Predict some training images to compute RMSE. These files don't exist yet.
            pre_filename = '{0}/Documents/CS446_ML_Final_Project/xray-super-res/xray_images/train_images_output_no_denoise/train_{1:05d}.png'.format(home, index)
            imPre = cv2.imread(pre_filename)[:, :, 0]
            pre[index-start, :] = np.reshape(imPre, (16384,))

        rmseVector = computeRMSE(pre, tru)
        rmse = np.sum(rmseVector)

        print("RMSE for {0} through {1}:\n\t{2:10.3f}".format(start, end, rmse))