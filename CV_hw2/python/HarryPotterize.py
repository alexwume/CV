import cv2
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
import matplotlib.pyplot as plt


#Write script for Q2.2.4
opts = get_opts()
#read in images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = cv2.cvtColor(hp_cover,cv2.COLOR_BGR2RGB)
cv_desk = cv2.cvtColor(cv_desk,cv2.COLOR_BGR2RGB)
hp_cover = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))
# print(hp_cover.shape)
# print(cv_cover.shape)

matches, locs1, locs2 = matchPics(cv_desk,cv_cover, opts)
pair1 = locs1[matches[:,0]]
pair2 = locs2[matches[:,1]]
homography = computeH_ransac(pair1, pair2, opts)
# hp_warped = cv2.warpPerspective(hp_cover,homography,(cv_desk.shape[1],cv_desk.shape[0]))
# print(hp_warped)
# mask = compositeH(homography, cv_desk, hp_warped)
final_img = compositeH(homography, cv_desk, hp_cover)
# plt.imshow((1- mask)*(cv_desk)/255 + mask * hp_warped / 255)
plt.imshow(final_img)
plt.show()

