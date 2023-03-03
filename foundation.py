import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_image():
    """
    // 图片处理
    :return:
    """
    img = cv2.imread("image/0.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.imwrite('image/1.png', img)
    cv2.waitKey(100)
    cv2.destroyAllWindows()


def cv_video():
    """
    // 视频处理
    :return:
    """
    vi = cv2.VideoCapture('image/vdeo.mp4')
    open, frame = vi.read()
    while open:
        ret, frame = vi.read()
        if frame is None:
            break
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            try:
                if cv2.waitKey(10) & 0xFF == 27:  # 0xFF == 27 退出Esc代码
                    break
            except KeyboardInterrupt:
                print("用户手动终止")
                break
    vi.release()
    cv2.destroyAllWindows()


def cut_image():
    """
    // 图像截取
    :return:
    """
    img = cv2.imread('image/0.png', cv2.IMREAD_GRAYSCALE)
    cut = img[:100, :10]
    print(cut)
    cv2.imshow('cut_image', cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_pick_up():
    """
    // 颜色通道提取
    :return:
    """
    img = cv2.imread('image/0.png')
    b, g, r = cv2.split(img)  # 通道拆分
    print(b.shape, img.shape)

    img1 = cv2.merge((b, g, r))  # 通道合并
    print(img1.shape)
    # 只保留R通道 同理保留G通道&B通道
    cut_img = img.copy()
    cut_img[:, :, 0] = 0
    cut_img[:, :, 1] = 0
    cv2.imshow('cut_img_R', cut_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def boundary_fill():
    """
    // 边界填充
    :return:
    """
    img = cv2.imread('image/0.png')
    top_size, bottom_size, left_size, right_size = 10, 10, 10, 10
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT101)
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=255)
    plt.subplot(231), plt.imshow(img), plt.title('artwork ')  # 原图
    plt.subplot(232), plt.imshow(replicate), plt.title('replicate ')  # 复制法
    plt.subplot(233), plt.imshow(reflect), plt.title('reflect ')  # 反射法
    plt.subplot(234), plt.imshow(reflect101), plt.title('reflect101 ')  # 反射法
    plt.subplot(235), plt.imshow(wrap), plt.title('wrap ')  # 外包装法
    plt.subplot(236), plt.imshow(constant), plt.title('constant ')  # 常量法
    plt.show()


def image_fusion():
    """
    // 图像融合
    :return:
    """
    img_0 = cv2.imread('image/0.png')
    img_VC = cv2.imread('image/VC.png')
    img_00 = img_0 + 10  # 偏移

    plt.subplot(321), plt.imshow(img_0), plt.title("artwork")
    plt.subplot(322), plt.imshow(img_VC), plt.title("artwork")
    VC_and_0 = img_VC + img_0  # 图像融合1
    plt.subplot(323), plt.imshow(img_00), plt.title("shifting_10")
    plt.subplot(324), plt.imshow(VC_and_0), plt.title("and")

    VC_add_0 = cv2.add(img_0, img_VC)
    plt.subplot(325), plt.imshow(VC_add_0), plt.title("add")
    VC_add_weight_0 = cv2.addWeighted(img_0, 0.5, img_VC, 0.5, 0)
    plt.subplot(326), plt.imshow(VC_add_weight_0), plt.title("add_weight")
    plt.show()


def threshold():
    """
    // 阈值处理
    :return:
    """
    img = cv2.imread('image/0.png')
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # 超过阈值部分取最大值
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # 超过阈值部分取最大值的翻转
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)  # 大于阈值部分设为阈值，否则不变
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)  # 大于阈值部分不变，否则设为0
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)  # 大于阈值部分不变，否则设为0的翻转
    title = ['artwork', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
    imgs = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(imgs[i])
        plt.title(title[i])
        plt.xticks([]), plt.yticks([])  # 去掉坐标轴
    plt.show()


def smoothing():
    """
    // 图像的平滑处理
    :return:
    """
    img = cv2.imread('image/lenaNoise.png')
    cv2.imshow('img', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    #  均值滤波操作/简单平均卷积操作 ###
    blur = cv2.blur(img, (3, 3))
    cv2.imshow('blur', blur)
    cv2.waitKey(1000)
    #  方框滤波、基本和均值一样，可以选择归一化,不选择归一化容易越界（normalize=True，归一化操作）
    box = cv2.boxFilter(img, -1, (3, 3), normalize=False)
    cv2.imshow('box', box)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    # 高斯滤波高斯模糊的卷积核里的数量是满足高斯分布，相当于更重视中间
    aussian = cv2.GaussianBlur(img, (3, 5), 1)
    cv2.imshow('aussian', aussian)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # 中值滤波
    median = cv2.medianBlur(img, 5)
    cv2.imshow('median', median)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    # 结果集中展示
    res = np.hstack((img, blur, aussian, median))  # 对图像进行拼接、hstack横向拼接，vstack纵向拼接
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def corrode():
    """
    // 形态学-腐蚀操作(需要先进行二值化)
    :return:
    """
    img = cv2.imread('image/dige.png')
    cv2.imshow('img', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=2)
    cv2.imshow('erosion', erosion)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    pie = cv2.imread('image/pie.png')
    cv2.imshow('pie', pie)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    kernel1 = np.ones((10, 10), np.uint8)
    erosion1 = cv2.erode(pie, kernel1, iterations=1)
    erosion2 = cv2.erode(pie, kernel1, iterations=3)
    erosion3 = cv2.erode(pie, kernel1, iterations=4)
    res = np.hstack((erosion1, erosion2, erosion3))
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def expand():
    """
    // 形态学-膨胀操作
    :return:
    """
    img = cv2.imread('image/dige.png')
    cv2.imshow('img', img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    kernel = np.ones((3, 3), np.uint8)
    # 先进行腐蚀操作
    erosion = cv2.erode(img, kernel, iterations=1)
    # 在进行膨胀操作
    ex_erosion = cv2.dilate(erosion, kernel, iterations=1)
    res = np.hstack((img, erosion, ex_erosion))
    cv2.imshow('原图--腐蚀后--膨胀后', res)
    cv2.waitKey(10)
    cv2.destroyAllWindows()

    pie = cv2.imread('image/pie.png')
    kernel = np.ones((20, 20), np.uint8)
    ex_erosion1 = cv2.dilate(pie, kernel, iterations=1)
    ex_erosion2 = cv2.dilate(pie, kernel, iterations=2)
    ex_erosion3 = cv2.dilate(pie, kernel, iterations=6)
    res1 = np.hstack((ex_erosion1, ex_erosion2, ex_erosion3))
    cv2.imshow('res1', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def opening_closing():
    """
    // 开运算与闭运算
    // 开运算：先腐蚀在膨胀
    // 闭运算：先膨胀在腐蚀
    :return:
    """
    # // 开运算 //
    img = cv2.imread('image/dige.png')
    cv2.imshow('img', img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    kernel = np.ones((3, 3), np.uint8)
    # 先进行腐蚀操作 分步操作
    erosion = cv2.erode(img, kernel, iterations=1)
    # 再进行膨胀操作
    ex_erosion = cv2.dilate(erosion, kernel, iterations=1)

    # 直接进行开运算，一步到位
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    res = np.hstack((erosion, ex_erosion, opening))
    cv2.imshow('分步开运算：腐蚀后--膨胀后-：-一步开运算开运算', res)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    if (ex_erosion == opening).all():  # 检验分步开运算与一步开运算结果是否一致
        print(True)
    else:
        print(False)

    # // 闭运算 //
    # 先进行膨胀操作
    ex_erosion1 = cv2.dilate(img, kernel, iterations=1)
    # 再进行腐蚀操作
    erosion1 = cv2.erode(ex_erosion1, kernel, iterations=1)
    # 直接进行闭运算，一步到位
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    res1 = np.hstack((ex_erosion1, erosion1, closing))
    cv2.imshow('res1', res1)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    if (closing == erosion1).all():  # 检验分步闭运算与一步闭运算结果是否一致
        print(True)
    else:
        print(False)


def gradient():
    """
    // 形态学操作-梯度计算 = 膨胀 - 腐蚀
    :return:
    """
    img = cv2.imread('image/pie.png')
    kernel = np.ones((10, 10), np.uint8)
    gradient1 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('gradient1', gradient1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def top_black_hat():
    """
    // 顶帽&黑帽
    // 顶帽 = 原始图像 - 开运算的结果
    // 黑帽 = 闭运算 - 原始图像
    :return:
    """
    img = cv2.imread('image/dige.png')
    kernel = np.ones((3, 3), np.uint8)
    # 顶帽
    top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # 黑帽
    black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    res = np.hstack((top_hat, black_hat))
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    top_black_hat()
