import argparse
import numpy as np
from SubtractDominantMotion import SubtractDominantMotion
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.08, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
fig = plt.figure()
for i in range(1,seq.shape[2]):
    mask = SubtractDominantMotion(seq[:,:,i-1],seq[:,:,i], threshold, num_iters, tolerance)
    # print(mask)
    objects = np.where(mask == 1)
    plt.imshow(seq[:,:,i], cmap='gray')
    plt.axis('off')
    plt.axis('tight')

    for j in range(len(objects[1])):
        patch = patches.Circle((objects[1][j],objects[0][j]),radius = 2)
        ax = plt.gca()
        ax.add_patch(patch)

    plt.show(block=False)
    plt.pause(0.2)
    # plt.clf()
    if i == 30 or i==60 or i==90 or i ==120:
        print(i)
        fig.savefig('../result/ant_frame' + str(i) + '.png',bbox_inches = 'tight')
    ax.clear()

