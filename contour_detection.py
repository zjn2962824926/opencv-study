import cv2
import numpy as np


def detection():
    """
    // 图像轮廓检测
    :return:
    """
    img = cv2.imread('image/contours.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 图像二值化
    binarty, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # binarty 等同于 thresh
    draw_img = img.copy()
    res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)  # 传入参数分别为：传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return contours


def contour_feature():
    """
    // 轮廓特征
    :return:
    """
    cnt = detection()[0]  # 轮廓索引
    # 计算轮廓面积
    area = cv2.contourArea(cnt)
    girth = cv2.arcLength(cnt, True)
    print(area, girth)


def contour_approximation():
    """
    // 轮廓近似
    :return:
    """
    img = cv2.imread('image/contours2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    draw_img = img.copy()
    res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
    cv2.imshow('res', res)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    draw_img1 = img.copy()
    res1 = cv2.drawContours(draw_img1, [approx], -1, (0, 0, 255), 2)
    cv2.imshow('res1', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    contour_approximation()
