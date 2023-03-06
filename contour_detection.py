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
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    binarty, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # binarty 等同于 thresh
    draw_img = img.copy()
    res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 1)  # 传入参数分别为：传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detection()
