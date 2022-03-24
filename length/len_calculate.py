# -*- ecoding: utf-8 -*-
# @ModuleName: 岩心长度及RQD计算
# @Function: 
# @Author: Jack Chen

import os
import cv2 as cv
import numpy as np
import pandas as pd


def len_calculate(img_path, cruve_path):

    img = cv.imread(img_path)
    GrayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.array(GrayImage)
    height, width = img.shape
    wi_row = []
    for row in range(1, height + 1):
        a = 0
        for col in range(width):
            val = img[row - 1][col]
            if (val) != 0:
                a += 1
        if a !=0:
            wi_row.append(row)

    # print(wi_row)
    cen_row = int((wi_row[-1] - wi_row[0]) / 2)
    cen_line = img[cen_row]
    # print(len(cen_line))

    file = pd.read_csv(cruve_path)
    cruve_data = pd.DataFrame(file, columns=None)
    h, w = cruve_data.shape
    boun_poi = [0]
    for i in range(0, h):
        X = list(cruve_data.iloc[i, :])
        y = int(X[0] * cen_row**4 + X[1] * cen_row**3 + X[2] * cen_row**2 + X[3] * cen_row + X[4])
        boun_poi.append(y)
    boun_poi.append(width)
    # print(boun_poi)
    core_len_li = []
    for i in range(0,len(boun_poi) - 1):
        core_li = list(cen_line[boun_poi[i] : boun_poi[i + 1]])
        core_len = core_li.count(255)
        if core_len >= 5:
            core_len_li.append(core_len)
    print(filename,"岩心长度列表(像素): ",core_len_li)

if __name__ == '__main__':
   img_dir = ".\\cal img\\"
   cruve_dir = ".\\cal_cruve\\"
   file_list = os.listdir(img_dir)
   print(file_list)
   for filename in file_list:
       img_path = img_dir + filename
       csv_name = os.path.splitext(filename)[0] + ".csv"
       cruve_path = cruve_dir + csv_name
       len_calculate(img_path, cruve_path)