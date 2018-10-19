'''generate training data, number of classes:2
    LA&RA:1, LV&RV$RVS:2,
    with 3 layer networks, with data augmentation'''
import os,sys
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
#import PIL
#from PIL import Image
#import Image
import cv2


class  SearchFile(object):
    
    def __init__(self,path='.'):
        self._path=path
        self.abspath=os.path.abspath(self._path) # 默认当前目录
    
    
    def findfile(self,keyword,root,data_augmentaiton = False):
        filelist=[]
        small_size = 0 # number of images whose size are 600*512
        large_size = 0 #number of images whose size are 800*512
        huge_size =0
        image_num = 0
        outimage =0
        for root,dirs,files in os.walk(root):
            for name in files:
                fitfile=filelist.append(os.path.join(root, name))
                print("image name{0}" .format(name))
                #print("files{0}" .format(files))
                #print("filelist{0}" .format(filelist))
                #count number of image with size 600*512
                if 'jpg' in name:
                    img = io.imread(name,as_grey = True)
                    image_num = image_num+1
                    if img.shape[1] == 600:
                        small_size = small_size+1
                    elif img.shape[1] == 800:
                        large_size = large_size+1
                    elif img.shape[1] == 900:
                        huge_size = huge_size+1
                    else:
                        print('....................********......................')
                        outimage = outimage +1
                        print('size of iamge is {0}' .format(img.shape))
                
                #print(fitfile)
                #print(os.path.join(root, name))
        #print(filelist)
        print('...........................................')
        #shape=(len(train_files)
        image_height = 512
        image_width =100
        num_file = 6*small_size+8*large_size+9*huge_size
        print("num_files,{0},{1},{2},{3},total num {4},outimage{5}" .format(num_file,small_size,huge_size,large_size,image_num,outimage))
        num_test = int(num_file/5+1)
        print("num of files:{0},large files{2},small files{1}" .format(num_file,small_size,large_size))
        # gennerate traning set and testing set
        x_train = np.ndarray(shape = (num_file-num_test, image_height, image_width))
        y_train = np.ndarray(shape = (num_file-num_test,),dtype = int)
        x_test = np.ndarray(shape = (num_test, image_height, image_width))
        y_test = np.ndarray(shape = (num_test,),dtype = int)
        j = 0
        j_test =0
        j_train =0
        num_LRA = 0
        num_LRV = 0
        num_RVS = 0
        for i in filelist:
            if os.path.isfile(i):
                #print(i)
                # LA&RA
                if (keyword[0] in os.path.split(i)[1])or (keyword[1] in os.path.split(i)[1]):
                    #print(os.path.split(i)[1])
                    #print('yes!',i)    # 绝对路径
                    #img=io.imread(i,as_grey=True)
                    #img = cv2.resize(img, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                    #print(" image shape is {0}".format(img.shape))
                    #dataset[j] = img
                    #label[j] = 1
                    #io.imshow(img)
                    #plt.show()
                    img = io.imread(i,as_grey=True)
                    num = int(img.shape[1]/100) #number of images genrated from the original image
                    for number in range(num):
                        if j%5 == 0:
                            x_test[int(j_test)] = img[:,number*100:number*100+100]
                            y_test[int(j_test)] = 0
                            j = j+1
                            j_test=j_test+1
                        else:
                            x_train[j_train] = img[:,number*100:number*100+100]
                            y_train[j_train] =0
                            j = j+1
                            j_train = j_train+1
                        num_LRA = num_LRA +1
            # LV&RV
                elif (keyword[2] in os.path.split(i)[1]) or (keyword[3] in os.path.split(i)[1]):
                    #print('RVS')
                    img = io.imread(i,as_grey=True)
                    num = int(img.shape[1]/100) #number of images genrated from the original image
                    for number in range(num):
                        if j%5 == 0:
                            x_test[int(j_test)] = img[:,number*100:number*100+100]
                            y_test[int(j_test)] = 1
                            j = j+1
                            j_test=j_test+1
                        else:
                            x_train[j_train] = img[:,number*100:number*100+100]
                            y_train[j_train] =1
                            j = j+1
                            j_train = j_train+1
                        num_RVS= num_RVS +1
            # LVS&RVS
                elif (keyword[4] in os.path.split(i)[1]) or keyword[5] in os.path.split(i)[1]:
                    img = io.imread(i,as_grey=True)
                    num = int(img.shape[1]/100) #number of images genrated from the original image
                    for number in range(num):
                        if j%5 == 0:
                            x_test[int(j_test)] = img[:,number*100:number*100+100]
                            y_test[int(j_test)] = 1
                            j = j+1
                            j_test=j_test+1
                        else:
                            print("name{0}" .format(os.path.split(i)[1]))
                            x_train[j_train] = img[:,number*100:number*100+100]
                            y_train[j_train] =1
                            j = j+1
                            j_train = j_train+1
                        num_RVS= num_RVS +1
                                
        print("dataset {0},data type{1}".format(x_train.shape,type(x_train)))
        print("dataset {0},data type{1}".format(y_train.shape,type(y_train)))
        print("testset {0},data type{1}".format(x_test.shape,type(x_test)))
        print("testset {0},data type{1}".format(y_test.shape,type(y_test)))
        #np.savez('oct_image.npz', x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
        print("number of LA:{0},number of RVS:{1}" .format(num_LRA,num_RVS))
        #dnpz = np.load('oct_image.npz')
        #lst = dnpz.files
        #print("list {0}" .format(int(lst)))
        print("size of npz {0}, type of npz {1}" .format(dnpz[lst[1]].shape, type(dnpz[lst[1]])))

               

#else:
#print('......no keyword!')

    def __call__(self):
               #keyword=input('the keyword you want to find:')
        keyword = ["LA","RA","LV","RV","RVS","LVS"]
        #eyword = ["L0","L1","R2","R3","RV4","LV5"]
        root=self.abspath
        data_augmentaiton = False
        self.findfile(keyword,root,data_augmentaiton)  # 查找带指定字符的文件


if __name__ == '__main__':
    search = SearchFile()
    search()
