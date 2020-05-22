# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
import skimage
from skimage.viewer import ImageViewer
import os
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from matplotlib import cm
from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    # convert m to mm
    rad = rad * 1e3
    pxSize = pxSize * 1e3

    rows = np.arange(res[0])
    cols = np.arange(res[1])
    coord = np.meshgrid(cols, rows)

    x_values = coord[1].reshape(-1)
    y_values = coord[0].reshape(-1)

    x_values = (x_values - res[0] // 2) * pxSize
    y_values = (y_values - res[1] // 2) * pxSize
    z_values = (rad ** 2 - x_values**2 - y_values ** 2)

    # modified_z_values = z_values
    invalid = np.where(z_values < 0)
    z_values = np.where(z_values < 0, 0, z_values)

    z_values = np.sqrt(z_values)
    surface_normal = np.vstack((x_values - center[0], y_values - center[1], z_values - center[2])) / rad

    intensity = np.dot(light, surface_normal)
    intensity[invalid] = 0
    intensity = np.where(intensity < 0, 0, intensity)

    image = intensity.reshape(res)


    return image

def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    #determine img shapes by loading the first image
    image_files = 'input_1.tif'
    image_path = os.path.join(path, image_files)
    im = skimage.io.imread(image_path, dtype = np.uint16)

    #convert rgb to xyz space, and extract the y component
    im_xyz = skimage.color.rgb2xyz(im)
    im = im_xyz[:,:,1]

    #return image shape
    s = im.shape
    im = im.reshape(-1)

    P = im.shape[0]

    #store the first img
    I = np.zeros((7, P))
    I[0] = im

    #load other images
    for itr in range(2, 8):
        image_files = 'input_' + str(itr) + '.tif'
        image_path = os.path.join(path, image_files)
        im = skimage.io.imread(image_path)
        im_xyz = skimage.color.rgb2xyz(im)
        im = im_xyz[:,:,1].reshape(-1)
        I[itr - 1] = im


    #loading light source
    sources_path = 'sources.npy'
    L = np.load(os.path.join(path, sources_path)).T

    return I, L, s

def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    B = np.linalg.inv( L @ L.T) @ L @ I
    return B

def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''


    albedos = np.linalg.norm(B, axis=0)

    #prevent divided by zeros
    norm = np.copy(albedos)
    norm = np.where(norm == 0, 1, norm)

    normals = B / norm

    return albedos, normals

def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape((s[0], s[1], 3))


    #visualize albedos image
    x_space = np.arange(s[0])
    y_space = np.arange(s[1])
    Y, X = np.meshgrid(y_space, x_space)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, albedoIm[:,:], cmap='gray', linewidth=0, antialiased=False)
    plt.show()

    #visualize normal images

    #make sure the pixel values are in the range of 0 ~ 1

    normal_copy = np.copy(normalIm)
    normal_copy[:,:,0] -= min(normal_copy[:,:,0].reshape(-1))
    normal_copy[:,:,0] /= max(normal_copy[:,:,0].reshape(-1))
    normal_copy[:,:,1] -= min(normal_copy[:,:,1].reshape(-1))
    normal_copy[:,:,1] /= max(normal_copy[:,:,1].reshape(-1))
    normal_copy[:,:,2] -= min(normal_copy[:,:,2].reshape(-1))
    normal_copy[:,:,2] /= max(normal_copy[:,:,2].reshape(-1))

    view = ImageViewer(normal_copy)
    view.show()


    return albedoIm, normalIm

def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    #zx = derivative of g in x dir
    #zy = derivative of g in y dir
    zx = np.zeros(s)
    zy = np.zeros(s)

    var = 1e10

    normals[2] = np.where(abs(normals[2]) == 0, var, normals[2])

    zx[:,:] = - (normals[0] / normals[2]).reshape(s)
    zy[:,:] = - (normals[1] / normals[2]).reshape(s)

    surface = integrateFrankot(zx, zy)

    return surface

def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    s = surface.shape
    x_space = np.arange(s[0])
    y_space = np.arange(s[1])
    Y, X = np.meshgrid(y_space, x_space)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, -surface[:,:], cmap='coolwarm', linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':

    #1b
    source_light1 = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    source_light2 = np.array([1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])
    source_light3 = np.array([-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])

    center = np.array([0, 0, 0])
    rad = 0.75 * 1e-2
    pxSize = 7 * 1e-6
    res = np.array([3840, 2160])
    image1 = renderNDotLSphere(center, rad, source_light1, pxSize, res)
    image2 = renderNDotLSphere(center, rad, source_light2, pxSize, res)
    image3 = renderNDotLSphere(center, rad, source_light3, pxSize, res)


    fig = plt.figure()
    fig.suptitle('Sphere in different lightning sources')
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(image1, cmap = 'gray')
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(image2, cmap = 'gray')
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(image3, cmap = 'gray')
    plt.show()

    #1c
    images, lightnings, shapes = loadData("../data/")

    u, s, vh = svds(images)
    # h = np.reshape(vh[-1])
    print(s)

    #1e
    B = estimatePseudonormalsCalibrated(images, lightnings)
    albedos, normals = estimateAlbedosNormals(B)

    #1f


    albedos_img, normals_img = displayAlbedosNormals(albedos, normals, shapes)

    #1h
    normals_reshape = np.zeros((3, shapes[0] * shapes[1]))
    normals_reshape[0,:] = normals_img[:,:,0].reshape(-1)
    normals_reshape[1,:] = normals_img[:,:,1].reshape(-1)
    normals_reshape[2,:] = normals_img[:,:,2].reshape(-1)

    surface = estimateShape(normals_reshape, shapes)
    plotSurface(surface)

