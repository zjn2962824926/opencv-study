import cv2


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
    img = cv2.imread('image/contours.png')
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
    res1 = cv2.drawContours(draw_img1, [approx], -1, (0, 255, 255), 2)
    cv2.imshow('res1', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bounding_box():
    """
    // 外接矩形
    :return:
    """
    img = cv2.imread('image/contours.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[2]  # 轮廓索引
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 轮廓面积与边界矩形比
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    print(x, y, w, h)
    rect_area = w * h
    extent = float(area) / rect_area
    print(f'轮廓面积与边界矩形比:{round(extent, 2)}')


def circumcircle():
    """
    // 外接圆
    :return:
    """
    img = cv2.imread("image/contours.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 图像二值化
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 轮廓提取
    cnt = contours[2]  # 轮廓选择
    # 外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)  # 该函数寻找包裹轮廓最小圆 radius：半径
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)  # 在图像上绘制一个圆
    cv2.imshow('img', img)
    print(cv2.minEnclosingCircle(cnt))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    circumcircle()
