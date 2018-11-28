from datetime import datetime
import imageio
import numpy as np
from scipy import ndimage

# Testing simple low-res to high-res image converter (cubic spline interpolation).

img_fldr = 'xray_images/'

tst_fldr = 'test_images_64x64_denoised/'
trn_fldr_low = 'train_images_64x64/'
trn_fldr_high = 'train_images_128x128/'
trn_out_fldr = 'train_images_output/'
tst_out_fldr = 'test_images_output/'

start = 1
end = 3999

# Set flag for processing either training or test images
isTest = True

for index in range(start, end + 1):
	tst_or_trn = 'test' if isTest else 'train'
	fname_in = '{0}_{1:05d}.png'.format(tst_or_trn, index)
	in_fldr = tst_fldr if isTest else trn_fldr_low
	path_low = img_fldr + in_fldr + fname_in
	
	# Read 64x64 image
	im = imageio.imread(path_low)

	# Interpolate to 128x128
	# print "im.shape = {0}".format(im.shape)
	im_interp = ndimage.zoom(im, [2., 2.])

	# Write 128x128 image
	fname_out = fname_in
	out_fldr = tst_out_fldr if isTest else trn_out_fldr
	path_out = img_fldr + out_fldr + fname_out
	imageio.imwrite(path_out, im_interp)