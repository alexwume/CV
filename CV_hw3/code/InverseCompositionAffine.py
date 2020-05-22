import numpy as np
from scipy.interpolate import RectBivariateSpline
#
def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()


    inter_spline_It1 = RectBivariateSpline(np.arange(It.shape[0]),np.arange(It.shape[1]),It1)
    inter_spline_It = RectBivariateSpline(np.arange(It.shape[0]),np.arange(It.shape[1]),It)
    x = np.arange(It.shape[1])
    y = np.arange(It.shape[0])
    x_, y_ = np.meshgrid(x, y)

    gradient_It_x = inter_spline_It.ev(y_,x_,dx = 0, dy = 1).flatten()
    gradient_It_y = inter_spline_It.ev(y_,x_,dx = 1, dy = 0).flatten()

    # print('gradient shape',gradient_It_y.shape)

    #gradient_T @ derivative of W :  steepest descent image
    steepest = np.zeros((gradient_It_x.shape[0], 6))
    steepest[:,2] = gradient_It_x
    steepest[:,5] = gradient_It_y

    for i in range (int(num_iters)):
        print(i)

        x1_ = p[0] * x_ + p[1] * y_ + p[2]
        y1_ = p[3] * x_ + p[4] * y_ + p[5]

        cond = (x1_ > 0) & (x1_ < It.shape[1]) & (y1_>0) & (y1_<It.shape[0])

        x1_ = x1_[cond]
        y1_ = y1_[cond]

        A_valid = steepest[cond.flatten()]

        inter_It1 = inter_spline_It1.ev(y1_,x1_)

        error_img = inter_It1.flatten() - It[cond].flatten()

        gradient_p = np.linalg.pinv(A_valid) @ error_img

        M = np.vstack((p.reshape((2,3)),[0,0,1]))

        delta_M = np.array([[1+gradient_p[0], gradient_p[1], gradient_p[2]], [gradient_p[3], 1+ gradient_p[4], gradient_p[5]],[0,0,1]])

        M = M @ np.linalg.pinv(delta_M)

        p = M[:2,:].flatten()

        if (np.sum(gradient_p **2) < threshold):
            break


    M = p.reshape((2,3))

    return M
