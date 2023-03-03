import cv2
import numpy as np


# //Canny 边缘检测流程//
# 1、使用高斯滤波器，以平滑图像，滤除噪声
# 2、计算图像中每个像素点的梯度强度和方向
# 3、应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散效应
# 4、应用双阈值（Double-Threshold）检测来确定真实的和潜在边缘
# 5、通过抑制孤立的弱边缘最终完成边缘检测

def cc():
    """
    // Canny边缘检测
    :return:
    """
    img = cv2.imread('image/lena.jpg', cv2.IMREAD_GRAYSCALE)
    canny1 = cv2.Canny(img, 80, 150)
    canny2 = cv2.Canny(img, 50, 100)
    res = np.hstack((canny1, canny2))
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cc()
