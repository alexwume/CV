import numpy as np
from skimage.color import rgb2gray
import numpy.matlib
import cv2
import numpy.ma as ma


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	N = x1.shape[0]
	A = np.zeros((2*N, 9))
	for i in range(N):
			A[ 2*i] = [x2[i][0], x2[i][1], 1,0,0,0,-x2[i][0]*x1[i][0], -x2[i][1]*x1[i][0], -x1[i][0]]
			A[ 2*i+1] = [0,0,0,-x2[i][0],-x2[i][1], -1,x2[i][0]*x1[i][1], x2[i][1]*x1[i][1], x1[i][1]]

	A = np.array(A)
	_, _, vh = np.linalg.svd(A)
	h = np.reshape(vh[8],(3,3))
	H2to1 = (1 / h[-1,-1]) * h

	return H2to1
def findCenter(xlist,ylist):
	center = np.zeros((1,2))
	xcenter = np.sum(xlist) / len(xlist)
	ycenter = np.sum(ylist) / len(ylist)
	center[0][0] = xcenter
	center[0][1] = ycenter
	return center
def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	#check what is x1, x2  here I assume to matching points

	x1_mean = findCenter(x1[:,0], x1[:,1])
	x2_mean = findCenter(x2[:,0], x2[:,1])


	# print('x1_mean = ', x1_mean)
	# print('x2_mean = ', x2_mean)
	#Shift the origin of the points to the centroid
	x1_shift = x1 - x1_mean
	x2_shift = x2 - x2_mean
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	# print('x1_shift: ',x1_shift)
	# x1_max = x1_shift[np.argmax(np.linalg.norm(x1_shift, axis = 1)) ]# / np.sqrt(2)
	# x2_max = x2_shift[np.argmax(np.linalg.norm(x2_shift, axis = 1))] #/ np.sqrt(2)
	# print(x1_shift)
	# print(np.max(abs(x1_shift[:,0])))
	# print(np.max(abs(x1_shift[:,1])))
	# x1_xmax = np.max(abs(x1_shift[:,0]))
	# x1_ymax = np.max(abs(x1_shift[:,1]))
	# x2_xmax = np.max(abs(x2_shift[:,0]))
	# x2_ymax = np.max(abs(x2_shift[:,1]))
	# x1_shift[:,0] = x1_shift[:,0] / x1_xmax
	# x1_shift[:,1] = x1_shift[:,1] / x1_ymax
	# x2_shift[:,0] = x2_shift[:, 0] / x2_xmax
	# x2_shift[:,1] = x2_shift[:, 1] / x2_ymax


	x1_norm = np.max(np.linalg.norm(x1_shift, axis = 1) / np.sqrt(2))
	x2_norm = np.max(np.linalg.norm(x2_shift, axis = 1) / np.sqrt(2))

	x1_normalized = (x1_shift * 1 / x1_norm)
	x2_normalized = (x2_shift * 1 / x2_norm)
	# print(np.linalg.norm(x1_normalized, axis = 1))

	# print('x1_norm',x1_norm)
	# print('x2_norm',x2_norm)
	# print('x1_normalized ', x1_normalized)
	# print('x2_normalized ', x2_normalized)
	#Similarity transform 1
	T1 = np.zeros((3,3))
	T1 [0,:] = [1/x1_norm, 0, -x1_mean[0][0] / x1_norm]
	T1 [1,:] = [0, 1/x1_norm, -x1_mean[0][1] / x1_norm]
	T1 [2,:] = [0,0,1]
	# T1[0, :] = [1 / x1_xmax, 0, -x1_mean[0][0] / x1_xmax]
	# T1 [1,:] = [0, 1/x1_ymax, -x1_mean[0][1] / x1_ymax]
	# T1 [2,:] = [0,0,1]

	# print('T1',T1)
	# print('det T1:', np.linalg.det(T1))
	#Similarity transform 2
	T2 = np.zeros((3, 3))
	T2[0, :] = [1 / x2_norm, 0, -x2_mean[0][0] / x2_norm]
	T2[1, :] = [0, 1 / x2_norm, -x2_mean[0][1] / x2_norm]
	T2[2, :] = [0, 0, 1]
	# T2[0, :] = [1 / x2_xmax, 0, -x2_mean[0][0] / x2_xmax]
	# T2[1, :] = [0, 1 / x2_ymax, -x2_mean[0][1] / x2_ymax]
	# T2[2, :] = [0, 0, 1]

	# print('det T2:', np.linalg.det(T2))
	# print('T2:', T2)
	# print('T1 :', T1)
	# print('T2 :',T2)

	#Compute homography
	H2to1_norm = computeH(x1_normalized, x2_normalized)
	# H2to1_norm = computeH(x1_shift, x2_shift)
	#Denormalization
	H2to1_norm = np.matmul(np.matmul(np.linalg.inv(T1) , H2to1_norm), T2)
	H2to1_norm = H2to1_norm / H2to1_norm[2][2]
	return H2to1_norm
def computeH_ransac(locs1, locs2, opts):

	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	# pair1 = locs1[matches[:,0]]
	# pair2 = locs2[matches[:,1]]
	pair1 = np.array(locs1)
	pair1[:,[0,1]] = pair1[:,[1,0]] # swap row and column due to openCV convention
	pair2 = np.array(locs2)
	pair2[:, [0, 1]] = pair2[:, [1, 0]] # swap row and column due to openCV convention
	rand_pair1 = np.zeros((4,2))
	rand_pair2 = np.zeros((4,2))

	max_count = 0
	bestH2to1 = np.eye(3)

	# transform to homogeneous coordinates
	homo_pair1 = np.transpose(np.hstack((pair1, np.ones((pair1.shape[0], 1)))))
	homo_pair2 = np.transpose(np.hstack((pair2, np.ones((pair2.shape[0], 1)))))

	while max_iters > 0:
			ind = np.random.choice(locs1.shape[0], 4, replace=False)
			rand_pair1[:,:] = pair1[ind]
			rand_pair2[:,:] = pair2[ind]
			H2to1 = computeH_norm(rand_pair1, rand_pair2)

			#project locs2 to locs1
			projected_pair2 = np.dot(H2to1, homo_pair2)
			projected_pair2[2,:] = np.where(projected_pair2[2,:] == 0 , 1, projected_pair2[2,:])
			projected_pair2 = projected_pair2 * (1 / projected_pair2[2, :])   # divide by z
			dist = np.sum((homo_pair1[0:2] - projected_pair2[0:2]) ** 2, axis=0) ** 0.5


			inlier_count = dist[dist <= inlier_tol].size
			if inlier_count > max_count:
				max_count = inlier_count
				bestH2to1 = H2to1
			max_iters -= 1
	return bestH2to1
def compositeH(H2to1, template, img):
	# img = rgb2gray(img)
	# mask = np.zeros((template.shape),dtype = np.uint8)
	# for i in range (mask.shape[0]):
	# 	for j in range (mask.shape[1]):
	# 		if img[i][j] != 0 : mask[i][j] = 1
	#
	# composite_img = mask
	mask = np.ones((img.shape),dtype = np.uint8)
	mask_warped = cv2.warpPerspective(mask,H2to1,(template.shape[1],template.shape[0]))
	img_warped = cv2.warpPerspective(img,H2to1,(template.shape[1],template.shape[0]))
	composite_img = (1- mask_warped)*(template) + mask_warped * img_warped

	return composite_img


