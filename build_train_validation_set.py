'''
    this code is used for building the data set for training and testing.
    
    '''

'''generate training data, number of classes:2
    LA&RA:1, LV&RV$RVS:2,
    with 3 layer networks, with data augmentation'''
import os,sys,random
from skimage import io
import numpy as np
#import PIL
#from PIL import Image
#import Image
import cv2
import shutil


class  SearchFile(object):
    
    def __init__(self,path='.'):
        self._path=path
        self.abspath=os.path.abspath(self._path) # 默认当前目录
    
    
    def findfile(self,dir1, dir2 ,root, ratio,path_abs):
        filelist=[]
        for root,dirs,files in os.walk(root):
            for name in files:
                #print(name)
                if '.jpg' in name:
                    #print(name)
                    fitfile=filelist.append(os.path.join(root, name))
        from random import shuffle
        shuffle(filelist)
        # print(filelist)
        num = round(len(filelist) * ratio)
        to_dir1, to_dir2 = filelist[:num], filelist[num:]
        for d in dir1, dir2:
            pathd= os.path.join(path_abs,d)
            if not os.path.exists(pathd):
                os.mkdir(pathd)
        print(os.path.join( path_abs,dir1))
        for file in to_dir1:
            shutil.copy2(file, os.path.join(dir1, os.path.basename(file)))
        for file in to_dir2:
            shutil.copy2(file, os.path.join(dir2, os.path.basename(file)))
    #else:
    #print('......no keyword!')

    def __call__(self):
        #keyword=input('the keyword you want to find:')
        #eyword = ["L0","L1","R2","R3","RV4","LV5"]
        root=self.abspath
        dir1 = 'data2/train'
        dir2 = 'data2/validation'
        ratio = 0.8
        path_abs = os.path.abspath('.')
        self.findfile(dir1, dir2 ,root, ratio,path_abs)  # 查找带指定字符的文件


if __name__ == '__main__':
    search = SearchFile()
    search()
