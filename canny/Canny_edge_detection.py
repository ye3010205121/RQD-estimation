# -*- ecoding: utf-8 -*-
# @ModuleName: Canny边缘检测
# @Function: 
# @Author: Jack Chen

import os
import cv2


def canny(img_path):
    img = cv2.imread(img_path, 0)
    edges = cv2.Canny(img, 80, 200)
    save_path = save_dir + img_path
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    cv2.imwrite(save_path, edges)


if __name__ == '__main__':
   #dir = input('please input the operate dir:')
   open_dir = "C:\\Users\\ZhangYe\\Desktop\\cannny_open\\"
   save_dir = "C:\\Users\\ZhangYe\\Desktop\\cannny_save\\"
   file_list = os.listdir(open_dir)
   print(file_list)
   for filename in file_list:
       img_path = dir + filename
       canny(img_path)
