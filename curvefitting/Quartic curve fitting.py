# -*- ecoding: utf-8 -*-
# @ModuleName: Quartic curve fitting
# @Function: 
# @Author: Jack Chen

import numpy as np
import os
import pandas as pd

def curve_fit(coord_path):
    print(coord_path)
    file = pd.read_csv(coord_path)
    data = pd.DataFrame(file, columns=None)
    X = list(data.iloc[0:, 0])
    Y = list(data.iloc[0:, 1])

    z1 = np.polyfit(X, Y, 4)
    p1 = np.poly1d(z1)
    print(p1)

if __name__ == '__main__':
   #dir = input('please input the operate dir:')
   dir = ".\\inter coords\\"
   file_list = os.listdir(dir)
   for filename in file_list:
       coord_path = dir + filename
       curve_fit(coord_path)

