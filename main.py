# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import cv2

import open3d as o3d

def points_to_image(xs, ys, img_size):
    coords = np.stack((xs, ys))
    abs_coords = np.ravel_multi_index(coords, img_size)
    img = np.bincount(abs_coords, weights=1, minlength=img_size[0]*img_size[1])
    img = img.reshape(img_size)
    return img


def linear_fit(x, y):
    pcd2 = o3d.geometry.PointCloud()
    x = np.array(x)
    y = np.array(y)
    results1 = sm.OLS(y, x).fit()
    results2 = sm.OLS(y, sm.add_constant(x)).fit()
    for i in range(pcd.points.__len__()):
        pcd2.points.append([x[i], results2.params[0] + x[i] * results2.params[1], 0])
    return pcd2

def cal_kb(p1, p2):
    k = p1[0] - p2[0] / p1[1] - p2[1]
    b = p1[1] - k * p1[0]
    return k, b

def dist_point_to_seg(p, k, b):
    return math.fabs(k * p[0] - p[1] + b) / math.pow(k * k + 1, 0.5)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("C:/Users/xvv/Desktop/oneSegment.ply")
    pcd.paint_uniform_color([0, 0, 1.0])
    x = []
    y = []
    for i in range(pcd.points.__len__()):
        x.append(pcd.points[i][0])
        y.append(pcd.points[i][1])
    pcd2 = linear_fit(x, y)
    o3d.visualization.draw_geometries([pcd2] + [pcd])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
