import numpy as np
import cv2
from opts import get_opts

#Import necessary functions
from tempfile import TemporaryFile
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from loadVid import loadVid
import matplotlib.pyplot as plt
#load videos to frames # run this once and store
# frames_panda = loadVid('../data/ar_source.mov')
# np.save('panda_frames', frames_panda)
# video_textbook = loadVid('../data/book.mov')
# np.save('bool_frames', video_textbook)


#Write script for Q3.1
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
# print('cv_cover shape', cv_cover.shape)
panda = np.load('../python/panda_frames.npy')
# print('panda shape',panda.shape)
book_frames = np.load('../python/bool_frames.npy')
# print('book shape',book_frames.shape)

#video cropping
frame_no = min(panda.shape[0],book_frames.shape[0])
image_ratio = cv_cover.shape[1] / cv_cover.shape[0]
edge_crop = 45
x_start = int(panda.shape[2]-((panda.shape[1]-2*edge_crop) * image_ratio))//2
x_end = int(x_start + ((panda.shape[1]-2*edge_crop)* image_ratio))

def createvideo(panda,book_frames,edge_crop, x_start,x_end,cv_cover):
    for idx in range(frame_no):
        panda_img = panda[idx]
        # crop the panda frame
        panda_img = panda_img[edge_crop:-edge_crop,x_start:x_end]
        panda_img = cv2.resize(panda_img, (cv_cover.shape[1], cv_cover.shape[0]))
        #book_frame
        book_img = book_frames[idx]

        matches, locs1, locs2 = matchPics(book_img, cv_cover, opts)
        pair1 = locs1[matches[:,0]]
        pair2 = locs2[matches[:,1]]
        homography = computeH_ransac(pair1, pair2, opts)
        # panda_warped = cv2.warpPerspective(panda_img,homography,(book_img.shape[1],book_img.shape[0]))
        # mask = compositeH(homography, book_img, panda_warped)
        merge_frame = compositeH(homography, book_img, panda_img)

        #merge frame
        # merge_frame = (1- mask)*(book_img) + mask * panda_warped
        # print(merge_frame.shape)
        ar_avi.write(merge_frame)

ar_avi = cv2.VideoWriter('../result/ar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))
createvideo(panda,book_frames,edge_crop, x_start,x_end,cv_cover)

ar_avi.release()
