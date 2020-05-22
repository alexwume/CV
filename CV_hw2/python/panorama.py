import cv2
from opts import get_opts

#Import necessary functions
from helper import plotMatches
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import imutils


#Write script for Q2.2.4
opts = get_opts()
#read in images
img_left = cv2.imread('../data/pano_left.jpg')
img_right = cv2.imread('../data/pano_right.jpg')
print(img_left.shape)
print(img_right.shape)
img_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB)
img_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB)


matches, locs1, locs2 = matchPics(img_left,img_right, opts)
pair1 = locs1[matches[:,0]]
pair2 = locs2[matches[:,1]]
homography = computeH_ransac(pair1, pair2, opts)
right_warped = cv2.warpPerspective(img_right,homography,(img_left.shape[1],img_left.shape[0]))

images = []
images.append(img_left)
images.append(img_right)

stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

if status == 0:
# write the output stitched image to disk
	# display the output stitched image to our screen
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
