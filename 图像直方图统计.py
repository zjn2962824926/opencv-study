import numpy as np
import matplotlib.pyplot as plt
import cv2


def test():
    """
    // 直方图定义
    :return:
    """
    # 图1
    # img = cv2.imread('image/car.png', 0)  # 0表示灰度图
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # print(hist.shape)
    # plt.hist(img.ravel(), 256)
    # plt.show()
    # 图2
    img1 = cv2.imread('image/car.png')
    colors = ['b', 'g', 'r']
    for i, col in enumerate(colors):
        histr = cv2.calcHist([img1], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
        plt.show()


def equalization():
    """
    // 均衡化原理
    :return:
    """


if __name__ == '__main__':
    test()
