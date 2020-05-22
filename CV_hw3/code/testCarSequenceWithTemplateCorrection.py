import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rect_init = np.copy(rect)
carseqrects_wcrt = np.zeros((seq.shape[2],4))
com = 0
update_th = 6
result_pre = np.zeros(2)
frame_0 = seq[:,:,0]
fig = plt.figure()
for i in range (1,seq.shape[2]):
    frame_now = seq[:,:,i]
    frame_com = seq[:,:,com]
    result_n = LucasKanade(frame_com,frame_now,rect,threshold,num_iters, p0 = np.zeros(2))
    result_star = LucasKanade(frame_0, frame_now,rect_init,threshold,num_iters, result_pre + result_n)


    if np.linalg.norm(result_star - result_pre - result_n) <= update_th:
        com = i
        result_pre = result_star
        rect = [rect_init[0] + result_star[0], rect_init[1] + result_star[1], rect_init[2] + result_star[0], rect_init[3] + result_star[1]]


    # print(np.linalg.norm(result_star - result_n))
    # print(com)
    # print(rect)
    # print(result_pre)



    # LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    plt.imshow(frame_now,cmap = 'gray')
    plt.axis('off')
    plt.axis('tight')
    patch = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] , rect[3] - rect[1] ,linewidth=1,edgecolor='r',facecolor='none')
    ax = plt.gca()
    ax.add_patch(patch)
    #visualizing the result
    plt.show(block=False)
    plt.pause(0.2)
    # if i == 1 or i==100 or i==200 or i ==300 or i ==400:
    #     fig.savefig('../result/carseq-wcrt_frame' + str(i) + '.png',bbox_inches = 'tight')
    carseqrects_wcrt[i,:] = np.array([rect[0], rect[1],rect[2], rect[3]])
    ax.clear()
np.save('../result/carseqrects-wcrt.npy',carseqrects_wcrt)

