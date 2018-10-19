'''
    this code is densighed to convert the mat into jpg
    '''
import scipy.io
import pandas as pd
import os,sys
from skimage import io
import numpy as np
import scipy.misc

#import PIL
#from PIL import Image
#import Image
import cv2
class Convert_mat_self(object):
    def __init__(self, path = '.'):
        self. path = path
        self.abspath = os.path.abspath(self.path) # the current path
    def convert_files(self, keyword, root):
        file_list = []
        num = 0
        for root, dirs, files in os.walk(root):
                for name in files:
                    filelist = file_list.append(os.path.join(root, name ))
                    if keyword  in name:
                        print(name)
                        mat = scipy.io.loadmat(name)['B_vol']
                        im = mat[:,:,1]
                        image_name = str(num)+ '.jpg'
                        print(image_name)
                        num +=1
                        print(mat.shape)
                        scipy.misc.toimage(im).save(image_name)
    # print('{0} has been saved' .format(image_name) )
    def __call__(self):
        root = self.abspath
        keyword = '.mat'
        self.convert_files(keyword, root)
if __name__ == '__main__':
    convert_mat_jpg = Convert_mat_self()
    convert_mat_jpg()









