import cv2
import numpy as np


def pyramid():
    """
    // 图像金字塔
    :return:
    """
    img = cv2.imread('image/AM.png')
    pyramid_up = cv2.pyrUp(img)  # 上采样
    cv2.imshow('pyramid_up', pyramid_up)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    pyramid_down = cv2.pyrDown(img)
    cv2.imshow('pyramid_down', pyramid_down)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    res = np.hstack((img, pyramid_up, pyramid_down))
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img.shape, pyramid_up.shape, pyramid_down.shape)
    if (cv2.pyrDown(pyramid_up) == img).all():
        print(True)
    else:
        print(False)


def lapkacian_pyr():
    """
    // 拉普拉斯金字塔
    // 原理：Li = Gi - pyrup(pyrdown(Gi))
    // Gi : 原图像
    // pyrup:向上采样   pyrdown:向下采样
    :return:
    """
    img = cv2.imread('image/AM.png')
    pyramid_down = cv2.pyrDown(img)
    pyramid_up = cv2.pyrUp(pyramid_down)
    lapkacian = img - pyramid_up
    cv2.imshow('lapkacian', lapkacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lapkacian_pyr()
