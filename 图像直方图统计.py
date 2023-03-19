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


def fourier_transform():
    """
    // 傅里叶变换
    // 频域&时域
    :return:
    """
    img = cv2.imread('image/lena.jpg', 0)
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # 得到灰度图能表示的形式
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title("input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def lpf_hpf():
    """
    // 低通滤波&高通滤波
    :return:
    """
    img = cv2.imread("image/lena.jpg", 0)
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, cool = int(rows / 2), int(cols / 2)  # 中心位置
    # 低通滤波器
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, cool - 30:cool + 30] = 1
    # 高通滤波器
    mask_hpf = np.ones((rows, cols, 2), np.uint8)
    mask_hpf[crow - 30:crow + 30, cool - 3:cool + 30] = 0
    # IDFT是dft的逆变换
    fshift = dft_shift * mask
    fshift_hpf = dft_shift * mask_hpf
    f_shift = np.fft.fftshift(fshift)
    f_shift_hpf = np.fft.fftshift(fshift_hpf)
    img_back = cv2.idft(f_shift)
    img_back_hpf = cv2.idft(f_shift_hpf)
    img_back1 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back1_hpf = cv2.magnitude(img_back_hpf[:, :, 0], img_back_hpf[:, :, 1])
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('artwork master'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back1, cmap='gray')
    plt.title('lpf_img'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back1_hpf, cmap='gray')
    plt.title('hpf_img'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    lpf_hpf()
