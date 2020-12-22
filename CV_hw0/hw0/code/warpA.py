import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor ."""

    #inverse of A
    A_inv=np.linalg.inv(A)

    #output image
    h_o=output_shape[0]
    w_o=output_shape[1]

    #construct position(i,j) of the output image
    uu,vv=np.meshgrid(np.arange(0,h_o),np.arange(0,w_o))
    arra1=np.linspace(1,1,uu.reshape(-1).shape[0])
    #stack three 1D array into a 3XN array
    uvpair=np.vstack([uu.reshape(-1),vv.reshape(-1),arra1])
    uvpair=uvpair.astype(int)

    #time with inverse of A to get the source pixel position
    xypair=np.dot(A_inv,uvpair)
    xypair=np.ndarray.round(xypair)
    xypair=xypair.astype(int)

    #get rid of the points that is out of the range of the source image
    xypair_xm=np.ma.masked_outside(xypair[0],0,h_o-1)
    xypair_ym=np.ma.masked_outside(xypair[1],0,w_o-1)

    mask_a=~xypair_xm.mask
    mask_b=~xypair_ym.mask
    mask_c=np.multiply(mask_a,mask_b)

    uvpair_x=uvpair[0]
    uvpair_y=uvpair[1]
    uvpair_x=uvpair_x[mask_c].data
    uvpair_y=uvpair_y[mask_c].data
    xypair_xm=xypair_xm[mask_c].data
    xypair_ym=xypair_ym[mask_c].data


    #construct result image
    result_image=np.zeros((h_o,w_o))
    result_image[uvpair_x,uvpair_y]=im[xypair_xm,xypair_ym]


    return result_image
