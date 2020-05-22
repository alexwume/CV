# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from scipy.sparse.linalg import svds
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u, s, vh = svds(images)
    u = u[:,-3:]
    # u = u[:,:3]

    #factorization 1
    s = s[-3:]
    # s = s[:3]
    s = np.sqrt(s)
    s_diag = np.eye(3)
    np.fill_diagonal(s_diag,s)

    vh = vh[-3:,:]
    # vh = vh[:3,:]

    B = u @ s_diag
    L = s_diag @ vh

    #factorization 2
    # s = s[-3:]
    # s_diag = np.eye(3)
    # np.fill_diagonal(s_diag,s)
    #
    # vh = vh[-3:,:]
    #
    # B = u
    # L = s_diag @ vh


    return B, L


if __name__ == "__main__":

    # Put your main code here

    images, lighting, shapes = loadData("../data/")

    #2b
    lightning_estimated, pseudo_normals = estimatePseudonormalsUncalibrated(images)

    albedos, normals = estimateAlbedosNormals(pseudo_normals)

    albedos_img, normals_img = displayAlbedosNormals(albedos, normals, shapes)


    for itr in range (6):
        lightning_estimated[itr,:] = lightning_estimated[itr,:] / np.linalg.norm(lightning_estimated, axis = 1)[itr]

    print("ground truth lighting\n", lighting.T)
    print("estimated lightning\n", lightning_estimated)


    #2d
    normals_reshape = np.zeros((3, shapes[0] * shapes[1]))
    normals_reshape[0,:] = normals_img[:,:,0].reshape(-1)
    normals_reshape[1,:] = normals_img[:,:,1].reshape(-1)
    normals_reshape[2,:] = normals_img[:,:,2].reshape(-1)
    surface_1 = estimateShape(normals_reshape, shapes)
    plotSurface(surface_1)

    #2e
    enforced_pseudo_normals = enforceIntegrability(pseudo_normals, shapes)

    lamb = 1
    v = 0.01   #increases v will decrease the difference along lateral axis
    mu = 0.01  #increase mu will decrease the depth difference among the faces
    G = np.asarray([[1,       0,   0],
                    [0,   1,    0],
                    [mu,   v, lamb]])

    enforced_pseudo_normals = G.T @ enforced_pseudo_normals



    albedos, normals = estimateAlbedosNormals(enforced_pseudo_normals)
    albedos_img, normals_img = displayAlbedosNormals(albedos, normals, shapes)

    normals_reshape = np.zeros((3, shapes[0] * shapes[1]))
    normals_reshape[0,:] = -normals_img[:,:,0].reshape(-1)
    normals_reshape[1,:] = -normals_img[:,:,1].reshape(-1)
    normals_reshape[2,:] = normals_img[:,:,2].reshape(-1)

    surface_2 = estimateShape(normals_reshape, shapes)
    plotSurface(surface_2)
