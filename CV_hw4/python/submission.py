"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import numpy.ma as ma
from scipy.optimize import leastsq

from helper import refineF
from helper import _singularize
from helper import displayEpipolarF
from helper import camera2
# Insert your package here

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    #TODO :: check  if 1 and 2 are not inversed!
    #normalize pts1 and pts2

    TT = np.eye(3)
    TT[0] = [2 / M, 0, -1]
    TT[1] = [0, 2 / M, -1]

    pts1_norm = TT @ np.vstack((pts1.T,np.ones(pts1.shape[0]))) #dim: 3 x N
    pts2_norm = TT @ np.vstack((pts2.T,np.ones(pts2.shape[0]))) #dim: 3 x N

    # eight point algorithm
    N = pts1.shape[0]
    U = np.ones((N,9))
    x1 = pts1_norm[0,:]
    y1 = pts1_norm[1,:]
    x2 = pts2_norm[0,:]
    y2 = pts2_norm[1,:]
    U[:,0] = x1*x2
    U[:,1] = x1*y2
    U[:,2] = x1
    U[:,3] = y1*x2
    U[:,4] = y1*y2
    U[:,5] = y1
    U[:,6] = x2
    U[:,7] = y2

    _,_,vh = np.linalg.svd(U)
    f = np.reshape(vh[8],(3,3))
    F = (1 / f[-1,-1]) * f

    #enforce singularity condition
    F = refineF(F, pts1_norm[:2].T, pts2_norm[:2].T)
    F = _singularize(F)

    F = TT.T @ F @ TT

    return F

def findCenter(list):
    center = np.zeros((1,2))
    center[0][0] = np.sum(list[:,0]) / len(list[:,0])
    center[0][1] = np.sum(list[:,1]) / len(list[:,1])

    return center

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):

    #pts1 and pts2 are 7 x 2
    TT = np.eye(3)
    TT[0] = [2 / M, 0, -1]
    TT[1] = [0, 2/M, -1]

    pts1_norm = TT @ np.vstack((pts1.T,np.ones(pts1.shape[0]))) #dim: 3 x N
    pts2_norm = TT @ np.vstack((pts2.T,np.ones(pts2.shape[0]))) #dim: 3 x N

    # seven point algorithm
    N = pts1.shape[0]
    U = np.ones((N,9))
    x1 = pts1_norm[0,:]
    y1 = pts1_norm[1,:]
    x2 = pts2_norm[0,:]
    y2 = pts2_norm[1,:]
    U[:,0] = x1*x2
    U[:,1] = x1*y2
    U[:,2] = x1
    U[:,3] = y1*x2
    U[:,4] = y1*y2
    U[:,5] = y1
    U[:,6] = x2
    U[:,7] = y2

    _,_,vh = np.linalg.svd(U)
    f1 = np.reshape(vh[-1],(3,3))
    f2 = np.reshape(vh[-2],(3,3))

    func = lambda x: np.linalg.det((1-x) * f1 + x * f2)
    d = func(0)
    c = 2 * (func(1) - func(-1)) / 3 - (func(2) - func(-2)) / 12
    b = (func(1) + func(-1)) / 2 - d
    a = (func(1) - func(-1)) / 2 - c
    x = np.roots([a, b, c, d])


    Fs = [(1 - x_) * f1 + x_ * f2 for x_ in x]
    Fs = [refineF(F, pts1_norm[:2].T, pts2_norm[:2].T) for F in Fs]
    Fs = [_singularize(F) for F in Fs]
    Fs = [TT.T @ F @ TT for F in Fs]

    return Fs


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    # pass
    E = K2.T @ F @ K1

    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # pass
    # http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    P1 = C1[0,:]
    P2 = C1[1,:]
    P3 = C1[2,:]
    P11 = C2[0,:]
    P22 = C2[1,:]
    P33 = C2[2,:]

    A = np.zeros((4,4))
    w = np.ones((pts1.shape[0],4))
    err = 0
    for i in range(pts1.shape[0]):
        # print(i)
        A[0] = y1[i] * P3 - P2
        A[1] = P1 - x1[i] * P3
        A[2] = y2[i] * P33 - P22
        A[3] = P11 - x2[i] * P33

        _,_,vh = np.linalg.svd(A)
        p = vh[-1,:]
        # print(p[-1])
        w[i,:3] = p[:3] / p[-1]

        # print(w[i])

        _projected1 = C1 @ w[i]
        _projected2 = C2 @ w[i]

        x1_projected = _projected1[0] / _projected1[2]
        y1_projected = _projected1[1] / _projected1[2]
        x2_projected = _projected2[0] / _projected2[2]
        y2_projected = _projected2[1] / _projected2[2]

        err = err + (x1[i] - x1_projected)**2 + (y1[i] - y1_projected)**2 + (x2[i] - x2_projected)**2 + (y2[i] - y2_projected)**2

    return w[:,:3], err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # pass

    x1 = int(round(x1))
    y1 = int(round(y1))
    # print(x1)
    # print(y1)

    win_size = 5
    delta = 40
    pts1_homo = np.hstack((x1,y1,1))
    line2 = F @ pts1_homo.T # 3x1
    y2 = np.arange(y1 - delta, y1 + delta)
    x2 = -(y2 * line2[1] + line2[2]) / line2[0]
    # print('x2',x2)
    # print('y2',y2)
    # print('len x2',len(x2))
    # print('len y2',len(y2))

    w = im2.shape[1]
    h = im2.shape[0]

    #gaussian weight
    gx, gy = np.meshgrid(np.linspace(-win_size, win_size, 2 * win_size + 1),np.linspace(-win_size, win_size, 2 * win_size + 1))
    d = np.sqrt(gx*gx + gy*gy)
    sigma, mu = 1, 0
    weight = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))

    if len(im1.shape) > 2:
        weight = np.repeat(np.expand_dims(weight, axis=2), im1.shape[-1], axis=2)
    # print('weight shape',weight.shape)

    cond = (x2 >= win_size)&(x2 < (w - win_size))&(y2 >= win_size )&(y2 < h - win_size)
    x2,y2 = x2[cond], y2[cond]

    x2 = np.ndarray.round(x2).astype(int)
    y2 = np.ndarray.round(y2).astype(int)

    print('len x2 after cond',len(x2))
    # print('len y2 after cond',len(y2))

    rect1 = [x1- win_size, y1 - win_size, x1 + win_size + 1, y1 + win_size + 1]
    # print('rect1',rect1)
    win1 = im1[rect1[1]:rect1[3],rect1[0]:rect1[2]]
    dist = 10000
    x2_ans = 0
    y2_ans = 0

    for i in range(x2.shape[0]):
        # x2[i] = int(x2[i])
        # y2[i] = int(y2[i])
        # print('x2[i]',x2[i])
        # print('y2[i]',y2[i])
        rect2 = [x2[i]- win_size, y2[i] - win_size, x2[i] + win_size + 1, y2[i] + win_size + 1]
        win2 = im2[rect2[1]:rect2[3],rect2[0]:rect2[2]]
        # print(rect2)
        # print('win1 shape',win1.shape)
        # print('win2 shape',win2.shape)
        dist_tmp = np.sum((win2-win1)**2 * weight)

        if dist_tmp < dist:
            dist = dist_tmp
            x2_ans = x2[i]
            y2_ans = y2[i]
    return x2_ans, y2_ans

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''

def ransacF(pts1, pts2, M, nIters, tol):
    rand_pair1 = np.zeros((7,2))
    rand_pair2 = np.zeros((7,2))
    max_iters = nIters
    max_count = 0
    best_F = None
    inliers = None

    pts1_norm = np.vstack((pts1.T,np.ones(pts1.shape[0]))) #dim: 3 x N
    pts2_norm = np.vstack((pts2.T,np.ones(pts2.shape[0]))) #dim: 3 x N

    while max_iters > 0:
        ind = np.random.choice(pts1.shape[0],7,replace = False)
        rand_pair1[:,:] = pts1[ind,:]
        rand_pair2[:,:] = pts2[ind,:]
        Fs = sevenpoint(rand_pair1, rand_pair2, M)

        for F in Fs:
            a, b = (F @ pts1_norm)[0:2]

            l2s = F @ pts1_norm/np.sqrt(a**2 + b**2)
            dist = abs(np.sum(pts2_norm * l2s, axis=0))

            inlier_count = dist[(dist < tol).T].shape[0]
            if inlier_count > max_count:
                max_count = inlier_count
                best_F = F
                inliers = (dist < tol).T

        max_iters -= 1
    return best_F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    r = r.reshape((3,1))
    theta = np.sqrt(np.sum(r**2))
    if theta == 0:
        ax = r
    else:
        ax = r / theta
    I = np.eye(3)

    # PS: ax[2] is type ndarray(slow), ax[2,0] is type float64(fast)
    ax_cross = np.array([[0, -ax[2, 0], ax[1, 0]], [ax[2, 0], 0, -ax[0, 0]], [-ax[1, 0], ax[0, 0], 0]])
    ax_cross_square = np.dot(ax, np.transpose(ax)) - np.sum(ax**2)*I

    R = I + np.sin(theta)*ax_cross + (1-np.cos(theta))*ax_cross_square

    return R
'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    ux = (R - R.T)
    u = np.array([ux[2,1], ux[0,2], ux[1,0]])
    u_l = np.sqrt(np.sum(u ** 2))
    tr = np.trace(R)

    # when l = 0 --> theta = 0 * n or pi * n
    # cos(theta) = (trace - 1) / 2
    if u_l == 0. and tr == 3: # theta = n * 0
        r = np.zeros((3,1))
        return r

    elif u_l == 0. and tr == -1.: # theta = n* pi
        tmp_a = R + np.diag(np.array([1, 1, 1]))
        v = None
        for i in range(3):
            if np.sum(tmp_a[:, i]) != 0:
                v = tmp_a[:, i]
                break
        tmp_b = v/np.sqrt(np.sum(v**2))
        r = np.reshape(tmp_b*np.pi, (3, 1))
        if np.sqrt(np.sum(r**2)) == np.pi and ((r[0, 0] == 0. and r[1, 0] == 0. and r[2, 0] < 0)
                                               or (r[0, 0] == 0. and r[1, 0] < 0) or (r[0, 0] < 0)):
            r = -r
        return r

    else:
        u = u / u_l
        theta = np.arctan2(np.float(u_l), np.float(tr) - 1)
        r = u * theta
        return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    # pass
    P, r2, t2 = x[:-6], x[-6:-3], x[-3:]
    N = P.shape[0] // 3
    P = np.reshape(P, (N, 3))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    P = np.vstack((P.T, np.ones(N)))

    p1_hat = K1 @ M1 @ P
    p1_hat = (p1_hat[:2, :] / p1_hat[2, :]).T
    p2_hat = K2 @ M2 @ P
    p2_hat = (p2_hat[:2, :] / p2_hat[2, :]).T

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    # pass
    # print('P_init shape', P_init.reshape(-1).shape)

    R2_init, t2_init = M2_init[:,:3], M2_init[:, 3]
    r2_init = invRodrigues(R2_init).reshape(-1)
    # print('r2 init', r2_init)
    # print(type(x))
    x0 = np.hstack((P_init.flatten(), r2_init.flatten(),t2_init.flatten()))
    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x) ** 2)

    x_update, _= leastsq(func, x0)


    P2, r2, t2 = x_update[:-6], x_update[-6:-3], x_update[-3:]
    N = P2.shape[0] // 3

    P2 = np.reshape(P2, (N, 3))
    r2 = np.reshape(r2, (3, 1))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    # pass


    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    x3 = pts3[:,0]
    y3 = pts3[:,1]

    P1 = C1[0,:]
    P2 = C1[1,:]
    P3 = C1[2,:]
    P11 = C2[0,:]
    P22 = C2[1,:]
    P33 = C2[2,:]
    P111 = C3[0,:]
    P222 = C3[1,:]
    P333 = C3[2,:]


    w = np.ones((pts1.shape[0],4))
    err = 0

    for i in range(pts1.shape[0]):
        con_val = np.array([pts1[i,-1], pts2[i,-1], pts3[i,-1]])
        N = sum(con_val >= Thres)
        if N < 2 : continue
        elif N == 2:
            A = np.zeros((4,4))
            A[0] = y1[i] * P3 - P2
            A[1] = P1 - x1[i] * P3
            A[2] = y2[i] * P33 - P22
            A[3] = P11 - x2[i] * P33
        else:
            A = np.zeros((6,4))
            A[0] = y1[i] * P3 - P2
            A[1] = P1 - x1[i] * P3
            A[2] = y2[i] * P33 - P22
            A[3] = P11 - x2[i] * P33
            A[4] = y3[i] * P333 - P222
            A[5] = P111 - x3[i] * P333


        _,_,vh = np.linalg.svd(A)
        p = vh[-1,:]
        # print(p[-1])
        w[i,:3] = p[:3] / p[-1]

        # print(w[i])

        _projected1 = C1 @ w[i]
        _projected2 = C2 @ w[i]
        _projected3 = C3 @ w[i]

        x1_projected = _projected1[0] / _projected1[2]
        y1_projected = _projected1[1] / _projected1[2]
        x2_projected = _projected2[0] / _projected2[2]
        y2_projected = _projected2[1] / _projected2[2]

        err_tmp =(x1[i] - x1_projected)**2 + (y1[i] - y1_projected)**2 + (x2[i] - x2_projected)**2 + (y2[i] - y2_projected)**2

        if N == 3:
            x3_projected = _projected3[0] / _projected3[2]
            y3_projected = _projected3[1] / _projected3[2]

            err_tmp = err_tmp + (x3[i] - x3_projected)**2 + (y3[i] - y3_projected)**2
        err = err + np.sqrt(err_tmp)

    return w[:,:3], err
