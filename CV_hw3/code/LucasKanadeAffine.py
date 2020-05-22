import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # p = M.flatten()
    p = np.zeros(6) #test
    p[0] = 1
    p[4] = 1

    inter_spline_It1 = RectBivariateSpline (np.arange(It1.shape[0]),np.arange(It1.shape[1]), It1)

    learning_rate = 0.5
    for i in range (int(num_iters)):



        # print(i)
        # pixel coordinates for the template
        x = np.arange(It.shape[1]-1 + 0.5)
        y = np.arange(It.shape[0]-1 + 0.5)
        x_, y_ = np.meshgrid(x, y)


        # pixel coordinates after affine transformation, i.e. pixel coordinates for It1
        x1_ = p[0] * x_ + p[1] * y_+ p[2]
        y1_ = p[3] * x_ +  p[4] * y_ + p[5]
        # print(x1_.shape)
        # print(y1_.shape)
        # when the pixel in It1 is not in It than we should not be comparing this pixel

        # print(max(x1_.flatten()))
        # print(max(y1_.flatten()))
        cond = (x1_ > 0) & (x1_ < It.shape[1]) & (y1_ > 0) & (y1_ < It.shape[0])
        x1_ = x1_[cond]
        # print(max(x1_.flatten()))
        # print(max(x1_))
        y1_ = y1_[cond]
        # print(max(y1_.flatten()))

        x_ = x_[cond]
        y_ = y_[cond]

        inter_It1 = inter_spline_It1.ev(y1_,x1_)
        #calculate the gradient a, It1
        gradient_I_x = inter_spline_It1.ev(y1_,x1_,dx = 0, dy = 1).flatten()
        gradient_I_y = inter_spline_It1.ev(y1_,x1_,dx = 1, dy = 0).flatten()

        #calculate A matrix
        A = np.zeros((gradient_I_x.shape[0], 6))
        A[:,0] = np.multiply(gradient_I_x, x_.flatten())
        A[:,1] = np.multiply(gradient_I_x, y_.flatten())
        A[:,2] = gradient_I_x
        A[:,3] = np.multiply(gradient_I_y, x_.flatten())
        A[:,4] = np.multiply(gradient_I_y, y_.flatten())
        A[:,5] = gradient_I_y


        gradient_p = np.linalg.pinv(A) @ (It[cond].flatten() - inter_It1.flatten())

        p = p + learning_rate * gradient_p.flatten()
        # if (np.sum(gradient_p **2) <  threshold):
        #     break
        if (np.linalg.norm(gradient_p) < threshold):
            break
    M = p.reshape((2,3))
    return M

##TEST======
# def LucasKanadeAffine(It, It1, threshold, num_iters):
#     """
#     #     :param It: template image
#     #     :param It1: Current image
#     #     :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
#     #     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
#     #     :param num_iters: number of iterations of the optimization
#     #
#     """
#     # put your implementation here
#     M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#     M = np.vstack((M,[0,0,1]))
#     p = np.zeros(6)
#
#
#     # inter_spline_It = RectBivariateSpline (np.arange(It.shape[0]),np.arange(It.shape[1]), It)
#     inter_spline_It1 = RectBivariateSpline (np.arange(It1.shape[0]),np.arange(It1.shape[1]), It1)
#
#
#     learning_rate = 1
#     for i in range (int(num_iters)):
#         print(i)
#         M [0,:] = [1+p[0], p[1], p[2]]
#         M [1,:] = [p[3], 1 + p[4], p[5]]
#
#         It1_affine = scipy.ndimage.affine_transform(It1, M[0:2,0:2], offset = M[0:2,2], output_shape=It.shape)
#
#         aff_mask = scipy.ndimage.affine_transform(np.ones(It1.shape),M, output_shape= It.shape)
#
#         # print(i)
#         # pixel coordinates for the template
#         x = np.arange(It.shape[1]-1 + 0.5)
#         y = np.arange(It.shape[0]-1 + 0.5)
#         x_, y_ = np.meshgrid(x, y)
#
#
#         # pixel coordinates after affine transformation, i.e. pixel coordinates for It1
#         x1_ = (1+p[0]) * x_ + p[1] * y_+ p[2]
#         y1_ = p[3] * x_+ (1+p[4]) * y_ + p[5]
#         # print(x1_.shape)
#         # print(y1_.shape)
#         # when the pixel in It1 is not in It than we should not be comparing this pixel
#
#         # print(max(x1_.flatten()))
#         # print(max(y1_.flatten()))
#         cond = (x1_ > 0) & (x1_ < It.shape[1]) & (y1_>0) & (y1_<It.shape[0])
#         x1_ = x1_[cond]
#         # print(max(x1_.flatten()))
#         # print(max(x1_))
#         y1_ = y1_[cond]
#         # print(max(y1_.flatten()))
#
#         x_ = x_[cond]
#         y_ = y_[cond]
#
#         inter_It1 = inter_spline_It1.ev(y1_,x1_)
#         #calculate the gradient a, It1
#         gradient_I_x = inter_spline_It1.ev(y1_,x1_,dx = 0, dy = 1).flatten()
#         gradient_I_y = inter_spline_It1.ev(y1_,x1_,dx = 1, dy = 0).flatten()
#
#         #calculate A matrix
#         A = np.zeros((gradient_I_x.shape[0], 6))
#         A[:,0] = np.multiply(gradient_I_x, x_.flatten())
#         A[:,1] = np.multiply(gradient_I_x, y_.flatten())
#         A[:,2] = gradient_I_x
#         A[:,3] = np.multiply(gradient_I_y, x_.flatten())
#         A[:,4] = np.multiply(gradient_I_y, y_.flatten())
#         A[:,5] = gradient_I_y
#
#
#         gradient_p = np.linalg.pinv(A) @ ((It *aff_mask).flatten() - It1_affine.flatten())
#
#         p = p + learning_rate * gradient_p.flatten()
#         # print(p)
#         if (np.sum(gradient_p **2) <  threshold):
#             break
#         # print(np.linalg.norm(gradient_p))
#         # if (np.linalg.norm(gradient_p) < threshold):
#         #     break
#     M = p.reshape((2,3))
#     return M
#test 2
# def LucasKanadeAffine(It, It1, threshold, num_iters):
#     """
#     :param It: template image
#     :param It1: Current image
#     :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
#     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
#     :param num_iters: number of iterations of the optimization
#     """
#
#     # put your implementation here
#     M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#     p = M.flatten()
#
#
#     inter_spline_It1 = RectBivariateSpline (np.arange(It1.shape[0]),np.arange(It1.shape[1]), It1)
#
#     learning_rate = 0.5
#     for i in range (int(num_iters)):
#         print(i)
#         M [0,:] = [1+p[0], p[1], p[2]]
#         M [1,:] = [p[3], 1 + p[4], p[5]]
#
#         # It1_affine = scipy.ndimage.affine_transform(It1, M[0:2,0:2], offset = M[0:2,2], output_shape=It.shape)
#         # It1_affine = cv2.warpPerspective(It1, M, (It.shape[1],It.shape[0]))
#         # aff_mask = scipy.ndimage.affine_transform(np.ones(It1.shape),M[0:2,0:2], offset = M[0:2,2], output_shape= It.shape)
#         # mask = np.ones((It1.shape), dtype = np.uint8)
#         # mask_affine = cv2.warpPerspective(mask, M, (It.shape[1],It.shape[0]))
#
#         # print(i)
#         # pixel coordinates for the template
#         x = np.arange(It.shape[1]-1 + 0.5)
#         y = np.arange(It.shape[0]-1 + 0.5)
#
#         x_, y_ = np.meshgrid(x, y)
#         x_ = x_.astype(int)
#         y_ = y_.astype(int)
#         # print(x_.shape)
#         # print(y_.shape)
#
#         # pixel coordinates after affine transformation, i.e. pixel coordinates for It1
#         x1_ = p[0] * x_ + p[1] * y_+ p[2]
#         y1_ = p[3] * x_+ p[4] * y_ + p[5]
#         x1_ = x1_.astype(int)
#         y1_ = y1_.astype(int)
#
#         # when the pixel in It1 is not in It than we should not be comparing this pixel
#         cond = (x1_ > 0) & (x1_ < It.shape[1]) & (y1_> 0) & (y1_ < It.shape[0])
#
#         # x_ = x_[cond]
#         # y_ = y_[cond]
#         x_ = x_[cond].flatten()
#         y_ = y_[cond].flatten()
#         x1_ = x1_[cond]
#         y1_ = y1_[cond]
#
#         inter_It1 = inter_spline_It1.ev(y1_,x1_)
#         #calculate the gradient a, It1
#         gradient_I_x = inter_spline_It1.ev(y_,x_,dx = 0, dy = 1).flatten()
#         gradient_I_y = inter_spline_It1.ev(y_,x_,dx = 1, dy = 0).flatten()
#
#         # print(gradient_I_x.shape)
#         # print(x_.flatten().shape)
#         #calculate A matrix
#         A = np.zeros((gradient_I_x.shape[0], 6))
#         A[:,0] = np.multiply(gradient_I_x, x_.flatten())
#         A[:,1] = np.multiply(gradient_I_x, y_.flatten())
#         A[:,2] = gradient_I_x
#         A[:,3] = np.multiply(gradient_I_y, x_.flatten())
#         A[:,4] = np.multiply(gradient_I_y, y_.flatten())
#         A[:,5] = gradient_I_y
#
#         # print(It1_affineatten().shape)
#         ## fix the mask method
#
#         # print(It1_affine.flatten().shape)
#         # print((It*mask_affine).flatten().shape)
#         # print(np.linalg.pinv(A).shape)
#         gradient_p = np.linalg.pinv(A) @ (It[cond].flatten() - inter_It1.flatten())
#         # print('done')
#         p = p + learning_rate * gradient_p.flatten()
#         if (np.sum(gradient_p **2) <  threshold):
#             break
#         # if (np.linalg.norm(gradient_p) < threshold):
#         #     break
#     M = p.reshape((2,3))
#     return M
