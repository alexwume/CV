import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None


    # print(image.shape)
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions


    ##########################
    ##### your code here #####
    ##########################
    sigma_est = skimage.restoration.estimate_sigma(image, multichannel = True, average_sigmas = True)
    # print(f"Estimated Gaussian noise standard deviation = {sigma_est}")
    denoise_img = skimage.restoration.denoise_tv_chambolle(image, weight = sigma_est, multichannel = True)
    # print('denoise img shape',denoise_img.shape)
    grey_img = skimage.color.rgb2gray(denoise_img)
    # print('grey img shape',grey_img.shape)
    thresh = skimage.filters.threshold_otsu(grey_img)
    # print('threshold',thresh)
    binary_img = grey_img < thresh
    # print('binary img shape',binary_img.shape)
    morph_img = skimage.morphology.opening(binary_img, skimage.morphology.square(3))
    bw = ~morph_img

    cleared = skimage.segmentation.clear_border(morph_img)
    label_img = skimage.measure.label(cleared, connectivity= 2)
    regions = skimage.measure.regionprops(label_img)
    mean_area = (np.sum([region.area for region in regions]) / len(regions)) / 2
    bboxes = [region.bbox for region in regions if region.area > mean_area]

    # for region in skimage.measure.regionprops(label_img):
    #     bboxes.append(region.bbox)
    #     print(region.bbox)
    return bboxes, bw
