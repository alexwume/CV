import numpy as np
from scipy.interpolate import RectBivariateSpline
import math
from scipy.ndimage import shift

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    #set rectangle window boundaries
    x_min = rect[0]
    y_min = rect[1]
    x_max = rect[2]
    y_max = rect[3]

    # print(x_min,x_max)
    # print(y_min, y_max)

    inter_spline_It = RectBivariateSpline(np.arange(It.shape[0]),np.arange(It.shape[1]),It)
    inter_spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]),np.arange(It1.shape[1]),It1)
    x_win = np.arange(x_min , x_max + 0.1 )
    y_win = np.arange(y_min , y_max + 0.1 )
    x_, y_ = np.meshgrid(x_win,y_win)
    inter_It = inter_spline_It.ev(y_,x_)

    learning_rate = 0.5
    # gradient_p = 1000000 # initialize gradient_p > threshold
    for i in range (int(num_iters)):
        x1_win = np.arange(x_min + p[0], x_max + p[0] + 0.1)
        y1_win = np.arange(y_min + p[1], y_max + p[1] + 0.1)
        x1_, y1_ = np.meshgrid(x1_win,y1_win)
        # print(x1_.shape)
        # print(y1_.shape)
        inter_It1 = inter_spline_It1.ev(y1_,x1_)
        # print(inter_It1.shape)

        gradient_I_x = inter_spline_It1.ev(y1_,x1_,dx = 0, dy = 1).flatten()
        gradient_I_y = inter_spline_It1.ev(y1_,x1_,dx = 1, dy = 0).flatten()

        # print('gradient I x shape',gradient_I_x.shape)

        A = np.zeros((gradient_I_x.shape[0], 2))
        A[:,0] = gradient_I_x
        A[:,1] = gradient_I_y

        # partial_w_p =
        # H = (gradient_I @ partial_w_p).T @ (gradient_I @ partial_w_p)
        # H = A.T @ A
        # T_x = inter_It.flatten()
        # I_wp = inter_It1.flatten()
        # gradient_p = np.linalg.inv(H) @ A.T @ (T_x - I_wp)
        # print('A shape',A.shape)
        # print('inter_It1 flatten',inter_It1.flatten().shape)
        # print('inter_It flatten',inter_It.flatten().shape)
        # print(inter_It1.shape)
        # print(inter_It.shape)

        gradient_p = np.linalg.pinv(A) @ (inter_It.flatten() - inter_It1.flatten())

        p = p + learning_rate * gradient_p.flatten()
        # print(np.sum(gradient_p**2))
        if (np.sum(gradient_p**2) <  threshold):
            break

    p[0] = round(p[0])
    p[1] = round(p[1])
    return p
