import numpy as np
import scipy.ndimage
import argparse
import numpy.ma as ma
import scipy.ndimage.morphology as mp
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    # mask = np.ones(image1.shape, dtype=bool)

    # M = LucasKanadeAffine(image1,image2, threshold, num_iters)

    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    M = np.vstack((M,np.array([0,0,1])))
    # print(M)

    #method 1
    image1_affine = scipy.ndimage.affine_transform (image1, M[0:2,0:2], offset = M[0:2,2], output_shape= image2.shape)
    aff_mask = scipy.ndimage.affine_transform(np.ones(image1.shape), M[0:2,0:2], offset = M[0:2,2], output_shape= image2.shape)
    # aff_mask = mp.binary_erosion(aff_mask,iterations=10)
    # image1_affine = mp.binary_erosion(image1_affine,iterations=20)
    diff = np.abs(image2 *aff_mask - image1_affine*aff_mask)
    # mask = ma.masked_greater (diff, tolerance).mask
    cond = diff > tolerance
    diff[cond] = 1
    diff[~cond] = 0

    mask = diff
    # print(mask.shape)
    mask = mp.binary_dilation(mask,iterations = 3)
    mask = mp.binary_erosion(mask, iterations = 1)


    # mask = scipy.ndimage.morphology.binary_erosion(mask)
    # mask = scipy.ndimage.morphology.binary_dilation(mask)
    return mask

#
# # if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
# parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
# parser.add_argument('--tolerance', type=float, default=0.4, help='binary threshold of intensity difference when computing the mask')
# args = parser.parse_args()
# num_iters = args.num_iters
# threshold = args.threshold
# tolerance = args.tolerance
# seq = np.load('../data/aerialseq.npy')
# mask = SubtractDominantMotion(seq[:,:,29],seq[:,:,30], threshold , num_iters, tolerance)
# print(np.sum(mask))
# objects = np.where(mask == 1)
# plt.figure()
# plt.imshow(seq[:,:,29], cmap='gray')
# plt.axis('off')
# fig,= plt.plot(objects[1],objects[0] ,'*')
# fig.set_markerfacecolor((0, 0, 1, 1))
# plt.show()
