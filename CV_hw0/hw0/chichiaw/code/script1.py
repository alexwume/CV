import os
import numpy as np
from PIL import Image
from alignChannels import alignChannels

# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
path = '/home/alex/Documents/CV_HW0/hw0/'
red = np.load(os.path.join(path, 'data/red.npy'))
green =  np.load(os.path.join(path, 'data/green.npy'))
blue = np.load(os.path.join(path, 'data/blue.npy'))
#blue =  np.load('/Users/apple/PycharmProjects/CV_HW0/hw0/data/blue.npy')


# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)

im = Image.fromarray(rgbResult, 'RGB')
im.save(os.path.join(path, "results/rgb_output.jpeg"))
