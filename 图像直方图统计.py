import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv_imshow


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
    # mask掩码操作
    img = cv2.imread('image/cat.jpg', 0)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    cv_imshow.show_img('mask', mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)  # 与操作
    cv_imshow.show_img('masked_img', masked_img)
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0, 256])
    plt.show()


def equalization_effect():
    """
    // 均衡化效果展示
    :return:
    """
    img = cv2.imread('image/cat.jpg', 0)
    plt.hist(img.ravel(), 256)
    plt.show()
    equ = cv2.equalizeHist(img)
    plt.hist(equ.ravel(), 256)
    plt.show()
    res = np.hstack((img, equ))
    cv_imshow.show_img('res', res, 10)
    # 自适应均衡化效果展示
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res_clahe = clahe.apply(img)
    res = np.hstack((img, equ, res_clahe))
    cv_imshow.show_img('res', res)


if __name__ == '__main__':
    equalization_effect()
