import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

#for image testing
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## =================testing ===============
# image = skimage.img_as_float(skimage.io.imread('../images/02_letters.jpg'))
# sigma_est = skimage.restoration.estimate_sigma(image, multichannel = True, average_sigmas = True)
# denoise_img = skimage.restoration.denoise_tv_chambolle(image, weight = sigma_est, multichannel = True)
# grey_img = skimage.color.rgb2gray(denoise_img)
# thresh = skimage.filters.threshold_otsu(grey_img)
# binary_img = grey_img < thresh
# morph_img = skimage.morphology.opening(binary_img, skimage.morphology.square(3))
# bw = ~morph_img
# cleared = skimage.segmentation.clear_border(morph_img)
# label_img = skimage.measure.label(cleared, connectivity= 2)
# regions = skimage.measure.regionprops(label_img)
# mean_area = (np.sum([region.area for region in regions]) / len(regions)) / 2
# bboxes = [region.bbox for region in regions if region.area > mean_area]
# np.save('../data/bboxes', bboxes)
# plt.imshow(bw, cmap = 'gray')
# np.save('../data/bw', bw)
# for bbox in bboxes:
#     minr, minc, maxr, maxc = bbox
#     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                             fill=False, edgecolor='red', linewidth=2)
#     plt.gca().add_patch(rect)
# plt.show()
#
# bboxes = np.load('../data/bboxes.npy')
# bw = np.load('../data/bw.npy')
# plt.imshow(bw,cmap = 'gray')
# # plt.show()

## ==================testing==============


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap = 'gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
#     plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    # print(bboxes)
    # print(len(bboxes))

    # Find the average characters size
    centers = [bbox for bbox in bboxes]
    centers = sorted(centers, key = lambda x:x[0])
    # print('centers',centers)

    sizes = [[bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in bboxes] # y, x
    mean_size = np.sum(sizes, axis = 0) / len(sizes)
    # print('mean size:', mean_size)
    # heights = [c[0] for c in centers]

    curr_height = (centers[0][0] + centers[0][2]) / 2 # the most left top char's y location
    rows = []
    row =[]

    for c in centers:
        if ((c[0] + c[2]) / 2) > (curr_height + mean_size[0]):
            row = sorted(row, key = lambda x: x[1])
            rows.append(row)
            row = [c]
            curr_height = (c[0] + c[2]) / 2
        else:
            row.append(c)
    #sort the last formed row and append it the to big list
    row = sorted(row, key = lambda x: x[1])
    rows.append(row)
    # print('how many lines', len(rows))
    # print('boxes after sort',rows)


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    crops = []
    tolerance = 24
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for row in rows:
        crop = []
        for coord in row:
            y2, x2, y1, x1 = coord
            y2 = np.max((0, y2 - tolerance))
            y1 = np.min((y1 + tolerance, im1.shape[0]))
            x2 = np.max((0, x2 - tolerance))
            x1 = np.min((x1 + tolerance, im1.shape[1]))
            # crop_img =  bw[y2 - tolerance: y1 + tolerance, x2 - tolerance : x1 + tolerance]
            crop_img =  bw[y2 : y1 , x2: x1]
            pad_row = 0
            pad_col = 0
            if (y1 - y2) < (x1 - x2):
                # pad_col = (x1 - x2) // 20
                pad_row = ((x1 - x2) - (y1 - y2)) // 2 + pad_col
            elif (y1 - y2) > (x1 - x2):
                # pad_row = (y1 - y2) // 20
                pad_col = ((y1 - y2) - (x1 - x2)) // 2 + pad_row

            crop_img = np.pad(crop_img, ((pad_row,),(pad_col,)), 'constant', constant_values= (1, 1))
            # plt.imshow(crop_img, cmap = 'gray')
            # plt.show()

            crop_img = skimage.transform.resize(crop_img, (500, 500))
            crop_img = skimage.exposure.adjust_gamma(crop_img,5)

            crop_img = skimage.transform.resize(crop_img, (200, 200))
            crop_img = skimage.exposure.adjust_gamma(crop_img,5)
            # plt.imshow(crop_img, cmap = 'gray')
            # plt.show()
            crop_img = skimage.transform.resize(crop_img, (32, 32))
            crop_img = skimage.morphology.erosion(crop_img, skimage.morphology.square(2))
            crop_img = skimage.exposure.adjust_gamma(crop_img,5)


            crop_img = crop_img.T
            crop.append(crop_img.flatten())
        crops.append(np.array(crop))

        #visualize the cropped images
        # print(crop_img)
        # plt.imshow(crop_img, cmap = 'gray')
        # print('crop image shape',crop_img.shape)
        # plt.show()


###=========================

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    ########################
    ##### your code here #####
    ##########################
    for data in crops:
        h1 = forward(data, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        string = []
        for i in range(probs.shape[0]):
            ind = np.argmax(probs[i,:])
            string.append(letters[ind])
        print(string)
