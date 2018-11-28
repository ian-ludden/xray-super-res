import imageio
import numpy as np

# Applies bilateral filter to denoise 64x64 images.

img_fldr = 'xray_images/'

tst_fldr = 'test_images_64x64/'
trn_fldr_low = 'train_images_64x64/'
trn_out_fldr = 'train_images_64x64_denoised/'
tst_out_fldr = 'test_images_64x64_denoised/'

start = 1
end = 1

# Set flag for processing either training or test images
isTest = True

for index in range(start, end + 1):
	tst_or_trn = 'test' if isTest else 'train'
	fname = '{0}_{1:05d}.png'.format(tst_or_trn, index)
	in_fldr = tst_fldr if isTest else trn_fldr_low
	path_low = img_fldr + in_fldr + fname
	
	# Read 64x64 image
	im = imageio.imread(path_low)


	# TODO: figure out how to use bilateral_approximation.py

	# Write 64x64 denoised image
	out_fldr = tst_out_fldr if isTest else trn_out_fldr
	path_out = img_fldr + out_fldr + fname
	imageio.imwrite(path_out, im_mag_01)