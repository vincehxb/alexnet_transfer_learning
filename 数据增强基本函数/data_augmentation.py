'''
主要做图片的预处理
1.统一图片的大小
2.
'''
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize

def ImageResize(img_mat,target_size=256):
    '''
    alexnet论文提到的方法：取最小的边为256,在取中间的一小块
    :param img_mat: numpy结果的图片数据
    :param target_size: 目标大小
    :return: resize后的图片
    '''
    return imresize(img_mat,(256,256,3))
    #先缩放小的边
    # imshape=list(img_mat.shape)
    # min_index,max_index=np.argmin(imshape[:2]),np.argmax(imshape[:2])
    # imshape[min_index]=target_size
    # im_=imresize(img_mat,imshape)
    # #在取中间的一块作为resize后的图片
    # mid_=imshape[max_index]//2
    # s_index=mid_-(target_size//2)
    # return im_[s_index:s_index+target_size,:,:] if max_index==0 else im_[:,s_index:s_index+target_size,:]
def CropImage(img_mat,expand_pram=10,pictype='train'):
    '''
    随机截取原图（256*256）为（227*227）
    :param img_mat: 原图
    :param expand_pram:扩大 expand_pram**2 倍图片
    :return:以列表形式返回扩展后的图片
    '''
    if pictype == 'train':
        pool=list(range(256-227))
        mask_w=np.random.choice(pool,expand_pram,replace=False)
        mask_h = np.random.choice(pool, expand_pram, replace=False)
        expan_img=[]
        for i in mask_w:
            for j in mask_h:
                expan_img.append(img_mat[i:i+227,j:j+227,:])
    else:
        pool=[0,29]#15
        expan_img=[img_mat[15:15+227,15:15+227,:]]
        for i in pool:
            for j in pool:
                expan_img.append(img_mat[i:i+227,j:j+227,:])
    return expan_img
def AlexEnlarge(img_mat,expand_pram=10,target_size=256,pictype='train'):
    '''
    对于不同大小的图片，调用这个函数后先resize为256*256
    再random crop以及水平翻转
    :param img_mat: 原图输入矩阵
    :param expand_pram: 扩展后的图片数量：2*(expand_pram^2),默认扩展200倍
    :return: 列表形式的扩展后的图片
    '''
    #同一图片的大小
    same_size_img=ImageResize(img_mat,target_size=target_size)
    #随机截取小patch
    crop_img=CropImage(same_size_img, expand_pram=expand_pram,pictype=pictype)
    #水平翻转
    hfilp_img=[img_[:,::-1,:] for img_ in crop_img]
    crop_img.extend(hfilp_img)
    return crop_img

def RandomCrop(img_mat,expand_pram=10):
    '''
    将大小不一的图片随机截取227*227大小的patch
    扩展的图片数量为 expand_pram^2
    :param img_mat: 图片矩阵
    :param expand_pram: 扩展参数
    :return: 列表形式的227*227大小的图片
    '''
    img_size=list(img_mat.shape)
    pool=[list(range(i-227)) for i in img_size[:2]]
    mask_w = np.random.choice(pool[0], expand_pram, replace=False)
    mask_h = np.random.choice(pool[1], expand_pram, replace=False)
    expan_img = []
    for i in mask_w:
        for j in mask_h:
            expan_img.append(img_mat[i:i + 227, j:j + 227, :])
    return expan_img
