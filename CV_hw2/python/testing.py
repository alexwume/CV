import cv2
from opts import get_opts
import numpy as np

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches
import matplotlib.pyplot as plt

matches = np.array([[0,0],[1,1],[2,2,],[3,3]])
#Write script for Q2.2.4
opts = get_opts()
#read in images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = cv2.cvtColor(hp_cover,cv2.COLOR_BGR2RGB)
cv_desk = cv2.cvtColor(cv_desk,cv2.COLOR_BGR2RGB)

x_desk = np.array([[240, 163, 577, 498],[194,493,485,190]]).T
x_cov = np.array([[0,0,349,349],[0,439,439,0]]).T


homography = computeH_ransac(x_desk, x_cov,opts)
print(homography)
x_desk_homo = np.array([[240, 163, 577, 498],[194,493,485,190],[1,1,1,1]],dtype=float)
x_cov_homo = np.array([[0,0,349,349],[0,439,439,0],[1,1,1,1]], dtype=float)
# projected_x_cover = homography @ x_cov_homo
# projected_x_cover = projected_x_cover * (1 / projected_x_cover[2, :])
# print(projected_x_cover[:2,:])

hp_warped = cv2.warpPerspective(cv_cover,homography,(cv_desk.shape[1],cv_desk.shape[0]))
plt.imshow (hp_warped)
plt.show()
