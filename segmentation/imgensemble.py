# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Time: 21-01-20 18:26
# Author: ZhangYe
# File: image2.py
import numpy as np
import cv2
from PIL import Image
import os
import sys
import shutil

from matplotlib import pyplot as plt
path='C:\\Users\\ZhangYe\\Desktop\\image_ensemble2\\ensemble\\'
path1='C:\\Users\\ZhangYe\\Desktop\\image_ensemble2\\labeleb5\\'
path2='C:\\Users\\ZhangYe\\Desktop\\image_ensemble2\\labelinceptionv2\\'
path3='C:\\Users\\ZhangYe\\Desktop\\image_ensemble2\\labelse154\\'
path4='C:\\Users\\ZhangYe\\Desktop\\image_ensemble2\\resnext\\'
path5='C:\\Users\\ZhangYe\\Desktop\\image_ensemble2\\seresnet\\'
def iou_com(path1, path2, path3, path4, path5):
    files1 = os.listdir(path1)
    files1 = np.sort(files1)
    files2 = os.listdir(path2)
    files2 = np.sort(files2)
    files3 = os.listdir(path3)
    files3 = np.sort(files3)
    files4 = os.listdir(path4)
    files4 = np.sort(files4)
    files5 = os.listdir(path5)
    files5 = np.sort(files5)
    #i=0
    for f1, f2, f3, f4, f5 in zip(files1, files2, files3, files4, files5):
        imgpath1 = path1 + f1
        imgpath2 = path2 + f2
        imgpath3 = path3 + f3
        imgpath4 = path4 + f4
        imgpath5 = path5 + f5
        
        
        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)
        img3 = cv2.imread(imgpath3)
        img4 = cv2.imread(imgpath4)
        img5 = cv2.imread(imgpath5)
        
        img=(img1*0.2 + img2*0.2 + img3 * 0.2 + img4 * 0.2 + img5 * 0.2)  
        cv2.imwrite("result.png", img)
        
        i = 1
        j = 1

#像素平均值=85.3

        img = Image.open("result.png")#读取照片
        width = img.size[0]#长度
        height = img.size[1]#宽度
        for i in range(0,width):#遍历所有长度的点
            for j in range(0,height):#遍历所有宽度的点
                data = (img.getpixel((i,j)))#打印该图片的所有点
        #print (type(data))
        #print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
        #print (data[0])#打印RGBA的r值
                if (data[0]>=60):#RGBA的r值大于170，并且g值大于170,并且b值大于170#85
               #print (data)
                       img.putpixel((i,j),(128,0,0))#则这些像素点的颜色改成大红色
                else:
                       img.putpixel((i,j),(0,0,0))
        img = img.convert("RGB")#把图片强制转成RGB
        file_name, file_extend = os.path.splitext(f1)
        dirpath = path 
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.png')
        img.save(dst)
        
                       
iou_com(path1, path2, path3, path4, path5)

