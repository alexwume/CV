import numpy as np
import cv2
import skimage.color
from skimage.color import rgb2gray
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

	#Convert Images to GrayScale
	img_1 = rgb2gray(I1)
	img_2 = rgb2gray(I2)

	#Detect Features in Both Images
	locs1=corner_detection(img_1,sigma)
	locs2=corner_detection(img_2,sigma)

	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(img_1, locs1)
	desc2, locs2 = computeBrief(img_2, locs2)
	#print(desc1.shape)
	#print(locs1.shape)


	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)
	#print(matches.shape)

	return matches, locs1, locs2
