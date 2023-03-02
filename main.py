import cv2
import matplotlib.pyplot as plt


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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 中值滤波


if __name__ == '__main__':
    smoothing()
