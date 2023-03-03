import cv2
import numpy as np


def sobel():
    """
    // sobel 算子Gx : -1   0   1       Gy:  -1  -2  -1            x：右边减左边
                     -2   0   2  * A        0   0   0  * A       y：下边减上边
                     -1   0   1             1   2   1
    // 计算梯度
    // 梯度计算建议，建议先分别计算水平和竖直方向梯度值后再通过权重求和，效果较好
    :return:
    """
    img = cv2.imread('image/pie.png', cv2.IMREAD_GRAYSCALE)
    # 计算水平方向梯度值,计算为算子范围内，左边减右边值，为负数时默认为0显示为黑，故需要在计算后去绝对值
    # 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 图像的像素点取值区间在（0,255）使用cv2.CV_64F方法后取值区间（0,255）进行拓展
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    res_x = np.hstack((sobel_x, sobel_x_abs))
    cv2.imshow('sobel_x and sobel_x_abs', res_x)
    cv2.waitKey(10)
    cv2.destroyAllWindows()

    # 计算竖直方向梯度值
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    res_y = np.hstack((sobel_y, sobel_y_abs))
    cv2.imshow('sobel_y and sobel_y_abs', res_y)
    cv2.waitKey(10)
    cv2.destroyAllWindows()

    # 分别计算x和y值后，再通过权重求和
    sobel_xy = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    cv2.imshow('sobel_xy', sobel_xy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 计算水平&竖直方向上的梯度值 直接计算（不建议使用）
    sobel_x_y = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    sobel_x_y_abs = cv2.convertScaleAbs(sobel_x_y)
    res_x_y = np.hstack((sobel_x_y, sobel_x_y_abs))
    cv2.imshow('res_x_y', res_x_y)
    cv2.waitKey(10)
    cv2.destroyAllWindows()


def scharr():
    """
    // Scharr 算子 Gx: -3   0   3                 Gy:   -3    -10    -3
                      -10  0   10  * A                  0      0     0  * A
                      -3   0   3                        3     10     3
    // 相比于sobel算子结果差异更敏感
    :return:
    """
    img = cv2.imread('image/lena.jpg', cv2.IMREAD_GRAYSCALE)
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_x_abs = cv2.convertScaleAbs(scharr_x)
    scharr_y_abs = cv2.convertScaleAbs(scharr_y)
    scharr_x_y = cv2.addWeighted(scharr_x_abs, 0.5, scharr_y_abs, 0.5, 0)
    cv2.imshow('scharr_x_y', scharr_x_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lapkacian():
    """
    // lapkacian 算子G：  0    1    0
                         1   -4    1
                         0    1    0
    // 对结果差异更敏感，同时对噪音点也较敏感
    :return:
    """
    img = cv2.imread('image/lena.jpg', cv2.IMREAD_GRAYSCALE)
    la = cv2.Laplacian(img, cv2.CV_64F)
    la_abs = cv2.convertScaleAbs(la)
    cv2.imshow('la_abs', la_abs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contrast():
    """
    // 算法结果对比
    :return:
    """
    img = cv2.imread('image/lena.jpg', cv2.IMREAD_GRAYSCALE)
    # sobel 算法
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_xy = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    # Scharr 算法
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_x_abs = cv2.convertScaleAbs(scharr_x)
    scharr_y_abs = cv2.convertScaleAbs(scharr_y)
    scharr_xy = cv2.addWeighted(scharr_x_abs, 0.5, scharr_y_abs, 0.5, 0)
    # lapkacian 算法
    la = cv2.Laplacian(img, cv2.CV_64F)
    la_abs = cv2.convertScaleAbs(la)
    gather = np.hstack((img, sobel_xy, scharr_xy, la_abs))
    cv2.imshow('gather', gather)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    contrast()
