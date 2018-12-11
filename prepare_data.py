import os
import cv2
import numpy as np
import h5py

DATA_PATH = "/home/iludden2/Documents/xray-super-res/"
BIG = "xray_images/"
SMALL = "xray_images_first100/"
LOW_RES_PATH = "train_images_64x64/"
HI_RES_PATH = "train_images_128x128/"
LOW_RES_DENOISED_PATH = "train_images_64x64_denoised/"

def prepare_training_data(isSmall):

  version = SMALL if isSmall else BIG
  # low_res_path = DATA_PATH + version + LOW_RES_PATH
  low_res_path = DATA_PATH + version + LOW_RES_DENOISED_PATH
  hi_res_path = DATA_PATH + version + HI_RES_PATH

  low_res_names = os.listdir(low_res_path)
  hi_res_names = os.listdir(hi_res_path)
  low_res_names = sorted(low_res_names)
  hi_res_names = sorted (hi_res_names)
  low_res_nums = low_res_names.__len__()
  hi_res_nums = hi_res_names.__len__()

  if (low_res_nums != hi_res_nums):
    raise ValueError('mismatch between number of data elements and labels')

  nums = low_res_nums
  data = np.zeros([nums,64,64,1])
  labels = np.zeros([nums,128,128,1])

  for i in range(low_res_nums):
    low_res_tensor = cv2.imread(low_res_path + low_res_names[i])
    hi_res_tensor = cv2.imread(hi_res_path + hi_res_names[i])
    low_res_tensor = (low_res_tensor[:,:,0].astype(float))/255
    hi_res_tensor = (hi_res_tensor[:,:,0].astype(float))/255

    # if(low_res_tensor.shape != (64,64)):
    #   raise ValueError('image'+ low_res_path + low_res_names[i] + 'not in proper resolution (64x64)')

    # if(hi_res_tensor.shape != (64,64)):
    #   raise ValueError('image'+ hi_res_path + hi_res_path[i] + 'not in proper resolution (128x128)')

    data[i,:,:,0] = low_res_tensor
    labels[i,:,:,0] = hi_res_tensor

  # return (np.reshape(data,(100,4096)),np.reshape(labels,(100,16384)))
  return (data, labels)


def write_hdf5(data, labels, output_filename):
  x = data.astype(np.float32)
  y = labels.astype(np.float32)

  with h5py.File(output_filename, 'w') as h:
    h.create_dataset('data', data=x, shape=x.shape)
    h.create_dataset('label', data=y, shape=y.shape)
    # h.create_dataset()


def read_training_data(file):
  with h5py.File(file, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label



if __name__ == "__main__":
   # data, label = prepare_training_data(True)
   # write_hdf5(data, label, "./train_first100.h5")
   data, label = prepare_training_data(False)
   write_hdf5(data, label, "./train_denoised.h5")
