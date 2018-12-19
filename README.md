# CS 446 Final Project
## X-Ray Image Denoising and Super-Resolution

## Ian Ludden (iludden2) and Adithya Murali (adithya5)

## Model Overview
Bilateral Filter -> Very Deep Super-Resolution (VDSR) convolutional network

## Preprocessing
We preprocess the low-res images with the bilateralFilter function from the OpenCV (cv2) Python library. We chose the bilateral filter after reading through some of the provided references. The filter parameters (depth = 5, sigma_color = 65.0, sigma_spatial = 15.0) were chosen manually by inspecting the denoised output images for ranges of parameters and visually comparing them to a high-res image down-sized to 64x64. 

## CNN Training and Validation
We then train a VDSR model (heavily based on https://github.com/GeorgeSeif/VDSR-Keras, which implements https://arxiv.org/abs/1511.04587) on the filtered low-res and (unfiltered) high-res training images. We use a 75%/25% train/test split and early stopping on the validation loss to avoid overfitting. 

## Other Models Considered
Our "inspiration" was that this worked far better than our implementations of SRCNN (https://arxiv.org/abs/1501.00092), EED/EES (https://arxiv.org/abs/1607.07680), and a couple of other simpler architectures. We briefly considered implementing the patch-based kNN super-resolution technique described in the following paper: https://doi.org/10.1109/CVPR.2004.1315043. However, the bilateral filter and VDSR approach was successful enough that we opted to try different bilateral filter parameters rather than implement an entirely new model in the last stages of the project. 
