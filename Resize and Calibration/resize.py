# coding = utf-8
from PIL import Image
import os

def convert(dir,width,height):
    file_list = os.listdir(dir)
    print(file_list)
    for filename in file_list:
        path = ''
        path = dir+filename
        im = Image.open(path)
        out = im.resize((640,480),Image.ANTIALIAS)  #(160,1280)
        print ("%s has been resized!"%filename)
        out.save(path)

if __name__ == '__main__':
   #dir = input('please input the operate dir:')
   dir = "C:\\Users\\ZhangYe\\Desktop\\resize\\24\\"
   convert(dir,480,640)

#C:\\Users\\YeZhang\\Desktop\\image-segmentation\\data\\dataset1\\images_prepped_test\\
