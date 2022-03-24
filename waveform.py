# -*- ecoding: utf-8 -*-
# @ModuleName: 波形图生成
# @Function: 
# @Author: Jack Chen
import os, glob
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def draw_waveform(dir):
    for file in os.listdir(dir):
        file1 = dir + '\\' + file
        (filepath, tempfilename) = os.path.split(file)
        file_name = tempfilename + '.png'
        for file2 in os.listdir(file1):
            file3 = file1 + '\\' + file2
            file4 = glob.glob(os.path.join(file3, "*.png"))
            for file_name in file4:
                (filepath1, tempfilename1) = os.path.split(file_name)
                img = cv.imread(file_name)
                GrayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = np.array(GrayImage)
                height, width = img.shape
                img_y = []
                for col in range(1, width + 1):
                    a = 0
                    b = 0
                    for row in range(height):
                        val = img[row][col - 1]
                        if (val) == 0:
                            a = a + 1
                        else:
                            b = b + 1
                    img_y.append(b)
                img_x = list(range(1, width + 1))

                matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
                # "r" 表示红色，ms用来设置*的大小
                plt.figure(figsize=(12.8, 6.4))
                plt.plot(img_x, img_y, "black", marker='.', ms=8)
                plt.xticks(rotation=45)
                plt.xlabel("列数")
                plt.ylabel("白色像素个数")
                plt.title("图像每列白色像素数")

                poly_save = 'pix_img'
                isExists = os.path.exists(poly_save)
                if not isExists:
                    os.makedirs(poly_save)
                save_name = poly_save + '\\' + tempfilename + '_' + str(file2) + '_' + tempfilename1
                plt.savefig(save_name)
                plt.close()

                data_save_csv = 'pix_csv'
                isExists = os.path.exists(data_save_csv)
                if not isExists:
                    os.makedirs(data_save_csv)
                pix_name = os.path.splitext(tempfilename1)[0]
                # save csv
                chss = [img_x, img_y]
                chss = np.transpose(chss)
                save_name_1 = data_save_csv + '\\' + tempfilename + '_' + str(file2) + '_' + pix_name + '.csv'
                mess = pd.DataFrame(chss, columns=None)
                mess.to_csv(save_name_1, index=False, header=False)


if __name__ == '__main__':
    dir = 'canny_core'
    draw_waveform(dir)
