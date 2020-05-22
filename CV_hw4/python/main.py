import numpy as np
import matplotlib.pyplot as plt

from submission import eightpoint
from submission import essentialMatrix
from submission import ransacF
from submission import triangulate
from helper import displayEpipolarF
from submission import rodrigues
from submission import invRodrigues
from submission import bundleAdjustment
from submission import essentialMatrix
from helper import epipolarMatchGUI
from helper import camera2
from helper import visualize_keypoints
from helper import plot_3d_keypoint
from submission import MultiviewReconstruction

if __name__ == '__main__':
    # for question 2
    img1 = plt.imread('../data/im1.png')
    img2 = plt.imread('../data/im2.png')
    # print('F',F)
    # print('img1 shape', img1.shape)
    # print('img2 shape', img2.shape)
    # print('M', M)


    # data = np.load('../data/some_corresp.npz')
    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # M = max(img1.shape[0], img1.shape[1])
    # F = eightpoint(pts1,pts2,M)

    #save F
    # np.savez('../result/q2_1.npz', F = F, M = M)
    #visualize the result
    # F = np.load('../result/q2_1.npz')['F']
    # print(F)
    # print(np.load('../result/q2_1.npz')['M'])
    # print(essentialMatrix(F))
    # displayEpipolarF(img1, img2, F)


    #question 4.1
    # img1 = plt.imread('../data/im1.png')
    # img2 = plt.imread('../data/im2.png')
    # F =np.load('../result/q2_1.npz')['F']
    # epipolarMatchGUI(img1,img2,F)

    # pts1, pts2 done by manually print out each points and hard code
    # pts1 = [[64, 135], [119, 205], [185, 346], [222, 370], [506, 231], [452, 120], [417, 144], [250, 231], [406, 240], [250, 169]]
    # pts2 = [[65, 122], [119, 169], [187, 358], [225, 384], [504, 196], [446, 124], [413, 149], [250, 209], [406, 212], [249, 168]]
    # pts1 = np.array(pts1)
    # pts2 = np.array(pts2)

    # save F, pts1, pts2
    # np.savez('../result/q4_1.npz', F = F, pts1 = pts1, pts2 = pts2)
    # data = np.load('../result/q4_1.npz')
    # print(data['F'])
    # print(data['pts1'])
    # print(data['pts2'])

    # # for question 5.1
    # img1 = plt.imread('../data/im1.png')
    # img2 = plt.imread('../data/im2.png')
    # # # print('F',F)
    # # # print('img1 shape', img1.shape)
    # # # print('img2 shape', img2.shape)
    # # # print('M', M)
    # #
    # #
    # data = np.load('../data/some_corresp_noisy.npz')
    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # M = max(img1.shape[0], img1.shape[1])
    # # F = eightpoint(pts1,pts2,M)
    #
    # # F, inliers = ransacF(pts1,pts2,M, 10, 0.5 )
    # #save F
    # # np.savez('../result/q2_1.npz', F = F, M = M)
    # #visualize the result
    # displayEpipolarF(img1, img2, F)

    #question 5.2
    # r = np.ones([3,1])
    # R = rodrigues(r)
    # assert R.shape == (3,3), 'rodrigues returns 3x3 matrix'
    # print(R)
    #
    # R = np.eye(3)
    # r = invRodrigues(R)
    # assert (r.shape == (3, )) | (r.shape==(3,1)), 'invRodriques return 3x1 vector'
    # print(r)


    # question 5.3
    # data = np.load('../data/some_corresp_noisy.npz')
    # K1 = np.load('../data/intrinsics.npz')['K1']
    # K2 = np.load('../data/intrinsics.npz')['K2']
    # F, inliers = ransacF(pts1,pts2, M, 100, 1)
    # # np.savez('../result/tmp', F = F, inliers = inliers)
    # # F = np.load('../result/tmp.npz')['F']
    # # inliers = np.load('../result/tmp.npz')['inliers']
    # # print(sum(inliers))
    # # displayEpipolarF(img1, img2, F)
    # pts1 = data['pts1'][inliers]
    # pts2 = data['pts2'][inliers]
    # # print('inliers',pts1)
    # E = essentialMatrix(F,K1,K2)
    #
    # M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    # M2s = camera2(E)
    # C1 = K1 @ M1
    #
    # best_err = np.finfo('float').max
    # for ind in range(4):
    #     M2_tmp = M2s[:, :, ind]
    #     C2_tmp = K2 @ M2_tmp
    #     w_tmp,err_tmp = triangulate(C1,pts1,C2_tmp,pts2)
    #
    #     # condition for the right M2: all the depths are positive!
    #     if err_tmp < best_err:
    #         # print(np.min(w_tmp[:, -1]))
    #         best_err = err_tmp
    #         w_init = w_tmp
    #         M2_init = M2_tmp
    #         # print(err_tmp)
    # print(best_err)
    # M2, w = bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, w_init)
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
    #     err2 += np.sqrt(np.sum((projected1 - pts1[i]) ** 2 + (projected2 - pts2[i]) ** 2))
    #
    # print(err2)

    #question 6.1
    image1 = plt.imread('../data/q6/cam1_time6.jpg')
    image2 = plt.imread('../data/q6/cam2_time6.jpg')
    image3 = plt.imread('../data/q6/cam3_time6.jpg')
    data = np.load('../data/q6/time6.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    pts3 = data['pts3']
    M1 = data['M1']
    M2 = data['M2']
    M3 = data['M3']
    K1 = data['K1']
    K2 = data['K2']
    K3 = data['K3']
    # visualize_keypoints(image3, pts3, 800)
    # print(pts1)
    # print(pts2)
    # print(pts3)
    C1 = K1 @ M1
    C2 = K2 @ M2
    C3 = K3 @ M3
    # print(pts2)
    w_3d, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 100)

    # print(pts1[pts1[:,-1] > 800])
    # a = pts2[:,-1]
    # print(np.sort(a))
    # w_3d,err = triangulate(C1,pts1,C2,pts2)
    print(err)
    plot_3d_keypoint(w_3d)
    np.savez('../result/q6_1', w = w_3d)
    # w = np.load('../result/q6.1.npz')['w']
    # print(w)
    #question 6.2
    #
    # connections_3d = [[0,1], [1,3], [2,3], [2,0], [4,5], [6,7], [8,9], [9,11], [10,11], [10,8], [0,4], [4,8], [1,5], [5,9], [2,6], [6,10], [3,7], [7,11]]
    # color_links = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,0,255),(255,0,255),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255)]
    # colors = ['blue','blue','blue','blue','red','magenta','green','green','green','green','red','red','red','red','magenta','magenta','magenta','magenta']
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(0,10):
    #     print(i)
    #     image1_path = '../data/q6/cam1_time' + str(i) + '.jpg'
    #     image1 = plt.imread(image1_path)
    #     image2_path = '../data/q6/cam2_time' + str(i) + '.jpg'
    #     image2 = plt.imread(image2_path)
    #     image3_path='../data/q6/cam3_time' + str(i) + '.jpg'
    #     image3 = plt.imread(image3_path)
    #     data_path = '../data/q6/time' + str(i) + '.npz'
    #     data = np.load(data_path)
    #
    #     pts1 = data['pts1']
    #     pts2 = data['pts2']
    #     pts3 = data['pts3']
    #     M1 = data['M1']
    #     M2 = data['M2']
    #     M3 = data['M3']
    #     K1 = data['K1']
    #     K2 = data['K2']
    #     K3 = data['K3']
    #
    #     C1 = K1 @ M1
    #     C2 = K2 @ M2
    #     C3 = K3 @ M3
    #     # print(pts2)
    #     pts_3d, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 400)
    #
    #     # plot_3d_keypoint(w_3d)
    #
    #     num_points = pts_3d.shape[0]
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # for j in range(len(connections_3d)):
    #     #     index0, index1 = connections_3d[j]
    #     #     xline = [pts_3d[index0,0], pts_3d[index1,0]]
    #     #     yline = [pts_3d[index0,1], pts_3d[index1,1]]
    #     #     zline = [pts_3d[index0,2], pts_3d[index1,2]]
    #     #     ax.plot(xline, yline, zline, color=colors[j])
    #     # np.set_printoptions(threshold=1e6, suppress=True)
    #     for j in range(num_points):
    #         xdot = [pts_3d[j,0]]
    #         ydot = [pts_3d[j,1]]
    #         zdot = [pts_3d[j,2]]
    #         ax.scatter3D(xdot, ydot, zdot, cmap='Greens')
    #
    #
    #     # plt.show()
    # for j in range(len(connections_3d)):
    #         index0, index1 = connections_3d[j]
    #         xline = [pts_3d[index0,0], pts_3d[index1,0]]
    #         yline = [pts_3d[index0,1], pts_3d[index1,1]]
    #         zline = [pts_3d[index0,2], pts_3d[index1,2]]
    #         ax.plot(xline, yline, zline, color=colors[j])
    # np.set_printoptions(threshold=1e6, suppress=True)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
