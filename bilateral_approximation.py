"""
bilateral_approximation.py
Fast Bilateral Filter Approximation Using a Signal Processing Approach in Python

Copyright (c) 2014 Jack Doerner
https://gist.github.com/jackdoerner/b81ad881c4064470d3c0#file-bilateral_approximation-py

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy
import math
import scipy.signal, scipy.interpolate

def bilateral_approximation(data, edge, sigmaS, sigmaR, samplingS=None, samplingR=None, edgeMin=None, edgeMax=None):
	# This function implements Durand and Dorsey's Signal Processing Bilateral Filter Approximation (2006)
	# It is derived from Jiawen Chen's matlab implementation
	# The original papers and matlab code are available at http://people.csail.mit.edu/sparis/bf/

	inputHeight = data.shape[0]
	inputWidth = data.shape[1]
	samplingS = sigmaS if (samplingS is None) else samplingS
	samplingR = sigmaR if (samplingR is None) else samplingR
	edgeMax = numpy.amax(edge) if (edgeMax is None) else edgeMax
	edgeMin = numpy.amin(edge) if (edgeMin is None) else edgeMin
	edgeDelta = edgeMax - edgeMin
	derivedSigmaS = sigmaS / samplingS;
	derivedSigmaR = sigmaR / samplingR;

	paddingXY = math.floor( 2 * derivedSigmaS ) + 1
	paddingZ = math.floor( 2 * derivedSigmaR ) + 1

	# allocate 3D grid
	downsampledWidth = math.floor( ( inputWidth - 1 ) / samplingS ) + 1 + 2 * paddingXY
	downsampledHeight = math.floor( ( inputHeight - 1 ) / samplingS ) + 1 + 2 * paddingXY
	downsampledDepth = math.floor( edgeDelta / samplingR ) + 1 + 2 * paddingZ

	gridData = numpy.zeros( (downsampledHeight, downsampledWidth, downsampledDepth) )
	gridWeights = numpy.zeros( (downsampledHeight, downsampledWidth, downsampledDepth) )

	# compute downsampled indices
	(jj, ii) = numpy.meshgrid( range(inputWidth), range(inputHeight) )

	di = numpy.around( ii / samplingS ) + paddingXY
	dj = numpy.around( jj / samplingS ) + paddingXY
	dz = numpy.around( ( edge - edgeMin ) / samplingR ) + paddingZ

	# perform scatter (there's probably a faster way than this)
	# normally would do downsampledWeights( di, dj, dk ) = 1, but we have to
	# perform a summation to do box downsampling
	for k in range(dz.size):
	
		dataZ = data.flat[k]
		if (not math.isnan( dataZ  )):
			
			dik = di.flat[k]
			djk = dj.flat[k]
			dzk = dz.flat[k]

			gridData[ dik, djk, dzk ] += dataZ
			gridWeights[ dik, djk, dzk ] += 1

	# make gaussian kernel
	kernelWidth = 2 * derivedSigmaS + 1
	kernelHeight = kernelWidth
	kernelDepth = 2 * derivedSigmaR + 1
	
	halfKernelWidth = math.floor( kernelWidth / 2 )
	halfKernelHeight = math.floor( kernelHeight / 2 )
	halfKernelDepth = math.floor( kernelDepth / 2 )

	(gridX, gridY, gridZ) = numpy.meshgrid( range( int(kernelWidth) ), range( int(kernelHeight) ), range( int(kernelDepth) ) )
	gridX -= halfKernelWidth
	gridY -= halfKernelHeight
	gridZ -= halfKernelDepth
	gridRSquared = (( gridX * gridX + gridY * gridY ) / ( derivedSigmaS * derivedSigmaS )) + (( gridZ * gridZ ) / ( derivedSigmaR * derivedSigmaR ))
	kernel = numpy.exp( -0.5 * gridRSquared )
	
	# convolve
	blurredGridData = scipy.signal.fftconvolve( gridData, kernel, mode='same' )
	blurredGridWeights = scipy.signal.fftconvolve( gridWeights, kernel, mode='same' )

	# divide
	blurredGridWeights = numpy.where( blurredGridWeights == 0 , -2, blurredGridWeights) # avoid divide by 0, won't read there anyway
	normalizedBlurredGrid = blurredGridData / blurredGridWeights;
	normalizedBlurredGrid = numpy.where( blurredGridWeights < -1, 0, normalizedBlurredGrid ) # put 0s where it's undefined

	# upsample
	( jj, ii ) = numpy.meshgrid( range( inputWidth ), range( inputHeight ) )
	# no rounding
	di = ( ii / samplingS ) + paddingXY
	dj = ( jj / samplingS ) + paddingXY
	dz = ( edge - edgeMin ) / samplingR + paddingZ 

	return scipy.interpolate.interpn( (range(normalizedBlurredGrid.shape[0]),range(normalizedBlurredGrid.shape[1]),range(normalizedBlurredGrid.shape[2])), normalizedBlurredGrid, (di, dj, dz) )