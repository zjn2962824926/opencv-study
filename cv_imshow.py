import cv2


def show_img(name, img, time=0):
    """
    // 显示图像
    :param name:名称
    :param img:图像
    :param time:图像显示时间默认为o
    :return:
    """
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    name = 'img'
    img = cv2.imread('image/cat.jpg')
    show_img(name, img)
