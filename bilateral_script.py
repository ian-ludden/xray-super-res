import imageio
import numpy as np
from skimage.restoration import (denoise_bilateral, estimate_sigma)
from skimage import img_as_float

# Applies bilateral filter to denoise 64x64 images.

img_fldr = 'xray_images/'

tst_fldr = 'test_images_64x64/'
trn_fldr_low = 'train_images_64x64/'
trn_out_fldr = 'train_images_64x64_denoised/'
tst_out_fldr = 'test_images_64x64_denoised/'

start = 1
end = 3999

# Set flag for processing either training or test images
isTest = True

for index in range(start, end + 1):
	tst_or_trn = 'test' if isTest else 'train'
	fname = '{0}_{1:05d}.png'.format(tst_or_trn, index)
	in_fldr = tst_fldr if isTest else trn_fldr_low
	path_low = img_fldr + in_fldr + fname
	
	# Read 64x64 image
	im = imageio.imread(path_low)
	im0 = (im[:,:,0]).astype(np.float64)

	# Figure out best params for denoise_bilateral
	# sigma_est = estimate_sigma(im0, multichannel=True, average_sigmas=True)
	# print "sigma_est = {0}".format(sigma_est)
	
	# sigmaColors = [9.0, 11.0, 13.0]
	# sigmaSpatials = [1, 2, 3]
	# for sigmaColor in sigmaColors:
	# 	for sigmaSpatial in sigmaSpatials:
	# 		im_bf = denoise_bilateral(im0, sigma_color=sigmaColor, sigma_spatial=sigmaSpatial, multichannel=False)

	# 		# Write 64x64 denoised image
	# 		fname = '{0}_{1:05d}_{2:4.2f}_{3:02d}.png'.format(tst_or_trn, index, sigmaColor, sigmaSpatial)
	# 		out_fldr = tst_out_fldr if isTest else trn_out_fldr
	# 		path_out = img_fldr + out_fldr + fname
	# 		imageio.imwrite(path_out, im_bf.astype(np.uint8))

	# Looks like best params are sigma_color=13.0, sigma_spatial = 1 or 2
	im_bf = denoise_bilateral(im0, sigma_color=13.0, sigma_spatial=1.5, multichannel=False)

	# Write 64x64 denoised image
	out_fldr = tst_out_fldr if isTest else trn_out_fldr
	path_out = img_fldr + out_fldr + fname
	imageio.imwrite(path_out, im_bf.astype(np.uint8))