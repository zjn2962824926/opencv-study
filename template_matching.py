import cv2
import matplotlib.pyplot as plt
import numpy as np


def template():
    """
    // 模板匹配
    :return:
    """
    img = cv2.imread('image/lena.jpg', 0)
    tem = cv2.imread('image/face.jpg', 0)
    h, w = tem.shape[:2]
    print(f'img.shape:{img.shape}')
    print(f'tem.shape:{tem.shape}')
    matchTemplate = ['cv2.TM_SQDIFF', 'cv2.TM_CCORR', 'cv2.TM_CCOEFF', 'cv2.TM_SQDIFF_NORMED', "cv2.TM_CCORR_NORMED",
                     'cv2.TM_CCOEFF_NORMED']  # 模板匹配可选函数类型 NORMED归一化操作（建议选择带归一化操作的函数或方法数据更可靠）
    res = cv2.matchTemplate(img, tem, cv2.TM_CCOEFF_NORMED)
    print(f'res.shape:{res.shape}')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(f'最小值:{min_val}\n最大值:{max_val}\n最小值位置:{min_loc}\n最大值位置:{max_loc}')


def templates():
    """
    // 多种不同函数匹配效果展示
    :return:
    """
    img = cv2.imread('image/lena.jpg', 0)
    tem = cv2.imread('image/face.jpg', 0)
    h, w = tem.shape[:2]
    methods = ['cv2.TM_SQDIFF', 'cv2.TM_CCORR', 'cv2.TM_CCOEFF', 'cv2.TM_SQDIFF_NORMED', "cv2.TM_CCORR_NORMED",
               'cv2.TM_CCOEFF_NORMED']
    for meth in methods:
        img2 = img.copy()
        # 匹配方法的真值
        method = eval(meth)
        # print(method)
        res = cv2.matchTemplate(img, tem, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #   如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # 画矩形
        cv2.rectangle(img2, top_left, bottom_right, 255, 2)
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        plt.subplot(122), plt.imshow(img2, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


def template_match():
    """
    // 模板匹配多个对象
    :return:
    """
    img = cv2.imread('image/mario.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tem = cv2.imread('image/mario_coin.jpg', 0)
    h, w = tem.shape[:2]
    res = cv2.matchTemplate(gray, tem, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    # 取匹配程度大于80%的坐标
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # *表示可选参数
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    template_match()
