import numpy as np
import cv2
from matchPics import matchPics
import skimage.color
from skimage.color import rgb2gray
from scipy import ndimage, misc
import matplotlib.pyplot as plt

from opts import get_opts
from helper import plotMatches


#Q2.1.6


#Read the image and convert to grayscale, if necessary
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
img_ori = rgb2gray(cv_cover)
iteration = 36
match_hist = np.zeros(iteration)

######################
# For histogram of the rotate matches

# for i in range(iteration):
# 	#Rotate Image
#     img_rotate = ndimage.rotate(img_ori, 10*i)
#
# 	#Compute features, descriptors and Match features
#     matches , locs1, locs2 = matchPics(img_ori,img_rotate , opts)
#
# 	#Update histogram
#     match_hist[i] = matches.shape[0]
#
# x = np.linspace(0,10*(iteration - 1), iteration)
# plt.bar(x,match_hist)
# plt.show()


######################
# For visualizing rotate images

#rotate image at 20 degree
img_rotate = ndimage.rotate(img_ori, 20)
matches , locs1, locs2 = matchPics(img_ori,img_rotate , opts)

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.axis('off')
skimage.feature.plot_matches(ax,img_ori,img_rotate,locs1,locs2,matches,matches_color='r',only_matches=True)
plt.show()

#rotate image at 60 degree
img_rotate = ndimage.rotate(img_ori, 60)
matches , locs1, locs2 = matchPics(img_ori,img_rotate , opts)

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.axis('off')
skimage.feature.plot_matches(ax,img_ori,img_rotate,locs1,locs2,matches,matches_color='r',only_matches=True)
plt.show()

#rotate image at 120 degree
img_rotate = ndimage.rotate(img_ori, 120)
matches , locs1, locs2 = matchPics(img_ori,img_rotate , opts)

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.axis('off')
skimage.feature.plot_matches(ax,img_ori,img_rotate,locs1,locs2,matches,matches_color='r',only_matches=True)
plt.show()