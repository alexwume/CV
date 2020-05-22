'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
#import modules
import numpy as np

#import functions
from submission import essentialMatrix
from submission import triangulate
from submission import eightpoint
from helper import camera2
import matplotlib.pyplot as plt
if __name__ == '__main__':


    #for question 3.2
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    K1 = np.load('../data/intrinsics.npz')['K1']
    K2 = np.load('../data/intrinsics.npz')['K2']
    # img1 = plt.imread('../data/im1.png')
    # img2 = plt.imread('../data/im2.png')

    # M = max(img1.shape[0], img1.shape[1])
    # F = eightpoint(pts1,pts2,M)
    F = np.load('../result/q2_1.npz')['F']

    E = essentialMatrix(F,K1,K2)

    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    M2s = camera2(E)
    # print(M1)
    # print(K1.shape)
    # print(M1.shape)
    C1 = K1 @ M1
    # C2s = np.zeros((3,4,4))
    # print('C2s',C2s[:,:,0])
    # print(K2)
    # print(M2s[:,:,0])
    # w = np.zeros(3)
    best_err = np.finfo('float').max
    for j in range(4):
        C2_tmp = K2 @ M2s[:,:,j]
        w_tmp,err_tmp = triangulate(C1,pts1,C2_tmp,pts2)
        # print(err)
        # print(err_tmp)
        # print(np.min(w_tmp[:,-1]))

        if err_tmp < best_err:
            best_err = err_tmp
            w = w_tmp
            M2 = M2s[:,:,j]

    C2 = K2 @ M2
    # print('M2',M2)
    np.savez('../result/q3_3.npz', M2 = M2, C2 = C2, P = w)


