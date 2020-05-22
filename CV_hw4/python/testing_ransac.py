"""
Check the dimensions of function arguments
This is *not* a correctness check
Written by Chen Kong, 2018.
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import submission as sub
import helper
# data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
#
# N = data['pts1'].shape[0]
M = 640

# 2.1
# F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
# helper.displayEpipolarF(im1, im2, F8)
# np.savez('q2_1.npz', F=F8, M=M)

# 2.2
# pts1 = np.array([[256,270],[162,152],[199,127],[147,131],[381,236],[193,290],[157,231]])
# pts2 = np.array([[257,266],[161,151],[197,135],[146,133],[380,215],[194,284],[157,211]])
# Farray = sub.sevenpoint(pts1, pts2, M)
# helper.displayEpipolarF(im1, im2, Farray[1])
# np.savez('q2_2.npz', F=Farray[1], M=M, pts1=pts1, pts2=pts2)

# 3.1
# intrinsic = np.load('../data/intrinsics.npz')
# K1, K2 = intrinsic['K1'], intrinsic['K2']
# E = sub.essentialMatrix(F8, K1, K2)
# print(E)
#
# # 4.1
# selected_pts1, selected_pts2 = helper.epipolarMatchGUI(im1, im2, F8)
#np.savez('q4_1.npz', F=F8, pts1=selected_pts1, pts2=selected_pts2)

# 5.1
noise_data = np.load('../data/some_corresp_noisy.npz')
# F, inliers = sub.ransacF(noise_data['pts1'], noise_data['pts2'], M, 100, 0.5)
# np.savez('../result/tmp.npz', F=F, inliers=inliers)
F = np.load('../result/tmp.npz')['F']
inliers = np.load('../result/tmp.npz')['inliers']
print(sum(inliers))
# helper.displayEpipolarF(im1, im2, F)
# F_compare = sub.eightpoint(noise_data['pts1'], noise_data['pts2'], M)
# helper.displayEpipolarF(im1, im2, F_compare)

# 5.2
# r = np.ones([3, 1])
# R = sub.rodrigues(r)
# assert R.shape == (3, 3), 'rodrigues returns 3x3 matrix'
#
# R = np.eye(3)
# r = sub.invRodrigues(R)
# assert (r.shape == (3, )) | (r.shape == (3, 1)), 'invRodrigues returns 3x1 vector'

# question 5.3
# data = np.load('../data/some_corresp_noisy.npz')
# K1 = np.load('../data/intrinsics.npz')['K1']
# K2 = np.load('../data/intrinsics.npz')['K2']
# F = np.load('../python/tmpnew.npz')['F']
# inliers = np.load('../python/tmpnew.npz')['inliers']
# F, inliers = ransacF(pts1,pts2, M, 10, 0.5)
# np.savez('../result/tmp', F = F, inliers = inliers)
# F = np.load('../result/tmp.npz')['F']
# inliers = np.load('../result/tmp.npz')['inliers']
# displayEpipolarF(img1, img2, F)
# print(pts)
# print(np.sum(inliers))
# pts1 = data['pts1'][inliers]
# pts2 = data['pts2'][inliers]
# # print('inliers',pts1)
# E = sub.essentialMatrix(F,K1,K2)
#
# M1 = np.hstack((np.eye(3),np.zeros((3,1))))
# M2s = helper.camera2(E)
# C1 = K1 @ M1
#
# best_err = np.finfo('float').max
# for ind in range(4):
#     M2_tmp = M2s[:, :, ind]
#     C2_tmp = K2 @ M2_tmp
#     w_tmp,err_tmp = sub.triangulate(C1,pts1,C2_tmp,pts2)
#     # w_tmp, err = sub.triangulate(C1, p1, C2_tmp, p2)
#
#     # condition for the right M2: all the depths are positive!
#     print(np.min(w_tmp[:,-1]))
#     # if np.min(w_tmp[:, -1]) > 0:
#     if err_tmp < best_err:
#         # print(np.min(w_tmp[:, -1]))
#         best_err = err_tmp
#         w_init = w_tmp
#         M2_init = M2_tmp
#         # print(err)
# M2, w = sub.bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, w_init)
# # np.savez('../result/q5.1', M2 = M2, w = w)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
#
# # original 3d points
# for idx in range(w_init.shape[0]):
#     xs = w_init[idx][0]
#     ys = w_init[idx][1]
#     zs = w_init[idx][2]
#     ax.scatter(xs, ys, zs, marker='.', color = 'b')
#
#     xs_opt = w[idx][0]
#     ys_opt = w[idx][1]
#     zs_opt = w[idx][2]
#     ax.scatter(xs_opt, ys_opt, zs_opt, marker='.', color = 'r')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
#
# #optimize 3d points
# C2 = K2 @ M2
# w_3d = np.hstack((w, np.ones((w.shape[0],1))))
# err2 = 0
# for i in range(pts1.shape[0]):
#     projected1 = C1 @ w_3d[i, :].T
#     projected2 = C2 @ w_3d[i, :].T
#     projected1 = np.transpose(projected1[:2] / projected1[-1])
#     projected2 = np.transpose(projected2[:2] / projected2[-1])
#     # compute error
#     err2 += np.sum((projected1 - pts1[i]) ** 2 + (projected2 - pts2[i]) ** 2)
#
# print(err2)
