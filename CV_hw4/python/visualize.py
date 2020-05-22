'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt


from submission import essentialMatrix
from submission import epipolarCorrespondence
from helper import camera2
from submission import triangulate

if __name__ == '__main__':

    #load from data
    data = np.load('../data/templeCoords.npz')
    x1 = data['x1']
    y1 = data['y1']

    #load camera intrinsic matrices
    K1 = np.load('../data/intrinsics.npz')['K1']
    K2 = np.load('../data/intrinsics.npz')['K2']

    #load images
    img1 = plt.imread('../data/im1.png')
    img2 = plt.imread('../data/im2.png')
    M = max(img1.shape[0],img1.shape[1])

    #load F, E
    F = np.load('../result/q2_1.npz')['F']
    E = essentialMatrix(F,K1,K2)

    x2 = np.zeros((len(x1),1))
    y2 = np.zeros((len(y1),1))
    #calculate correspondence x2, y2
    for idx in range(x1.shape[0]):
        # print(int(x1[idx]))
        x2[idx],y2[idx] = epipolarCorrespondence(img1, img2, F, int(x1[idx]), int(y1[idx]))

    #form pts1 and pts2
    pts1 = np.hstack((x1, y1))
    pts2 = np.hstack((x2, y2))

    #calculate M1, M2s, C1
    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    M2s = camera2(E)
    C1 = K1 @ M1

    #calculate M2
    for j in range(4):
        C2_tmp = K2 @ M2s[:,:,j]
        [w_tmp,err_tmp] = triangulate(C1,pts1,C2_tmp,pts2)
        if np.min(w_tmp[:,-1]) >= 0:
            err = err_tmp
            w = w_tmp
            M2 = M2s[:,:,j]
            break

    C2 = K2 @ M2

    # print('w',w)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    for idx in range(w.shape[0]):
        xs = w[idx][0]
        ys = w[idx][1]
        zs = w[idx][2]
        ax.scatter(xs, ys, zs, marker='.', color = 'b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


    np.savez('../result/q4_2.npz',M1 = M1, M2 = M2, C1 = C1, C2 = C2, F = F)
