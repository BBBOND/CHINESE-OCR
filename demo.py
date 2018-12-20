# coding:utf-8
import time
from glob import glob
import argparse

import numpy as np
from PIL import Image

import model
# ces

paths = glob('./test/*.*')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        default='./test/001.png',
        help='图片名'
    )
    parser.add_argument(
        '--adjust',
        type=bool,
        default=False,
        help='调整文字识别结果'
    )
    parser.add_argument(
        '--detectAngle',
        type=bool,
        default=False,
        help='是否检测文字朝向'
    )
    args, unparsed = parser.parse_known_args()

    print(args)

    im = Image.open(args.image)
    img = np.array(im.convert('RGB'))
    t = time.time()
    '''
    result,img,angel分别对应-识别结果，图像的数组，文字旋转角度
    '''
    result, img, angle = model.model(
        img, model='keras', adjust=args.adjust, detectAngle=args.detectAngle)
    print("It takes time:{}s".format(time.time() - t))
    print("---------------------------------------")
    for key in result:
        print(result[key][1])
