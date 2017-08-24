import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import os
import pickle
import data_augmentation as da
def GetFlowerPicAddr(Proot=r'D:\Data warehouse\5 flower\flower_photos',
                     trainimg_tate=0.8):
    '''
    img_dict={
    'roses':{'train':[addr],'test':[addr]},
    .....
    }
    :param Proot:
    :param trainimg_tate:
    :return:
    '''
    #得到分类好的花图片相对地址
    is_root = True
    img_dict = {}
    for root, dirlist, filelist in os.walk(Proot):
        if is_root:
            # 第一次得到的是当前目录
            is_root = False
            continue
        flow_type = root.split('\\')[-1]
        img_dict[flow_type] = {'train': [], 'test': []}
        num_img = len(filelist)
        num_train = int(trainimg_tate * num_img)
        num_test = num_img - num_train
        train_mask = np.random.choice(range(num_img), num_train, replace=False)
        test_mask = np.random.choice(list(set(range(num_img)) - set(train_mask)), num_test, replace=False)
        #将相对地址填充成绝对地址
        flower_root=os.path.join(Proot,flow_type)
        filelist_addr=[os.path.join(flower_root,addr_) for addr_ in filelist]
        filelist = np.asarray(filelist_addr)  # 方便采样
        img_dict[flow_type]['train'] = filelist[train_mask]
        img_dict[flow_type]['test'] = filelist[test_mask]
    return img_dict
def SaveTestImage(img_addr_dict):
    '''
    将测试图片的扩展后存储起来
    对于每个花的种类，假设原有测试图片N张，以（N，10,227,227,3）矩阵形式存储
    :param img_addr_dict:
    :return:
    '''
    test_img_dict={}
    flower=['dandelion','tulips','roses','sunflowers','daisy']
    for index,fname in enumerate(flower):
        img_list=np.zeros((1,227,227,3))
        print(fname)
        lable=[0]*5
        lable[index]=1
        test_img_dict[fname]={'image':None,'labels':None}
        #将每个花的测试集resize,放到字典里
        for addr_ in img_addr_dict[fname]['test']:
            img_=imread(addr_)[:,:,:3]
            rx=da.AlexEnlarge(img_,pictype='test')
            rx=np.asarray(rx).reshape((-1,227,227,3))
            img_list=np.vstack((img_list,rx))
        lables=lable*(img_list.shape[0]-1)
        test_img_dict[fname]['image']=img_list[1:,:,:,:].astype('uint8')
        test_img_dict[fname]['lables']=np.asarray(lables).reshape((-1,5)).astype('uint8')
    print('Load test image done!')
    fp=open(r'D:\Data warehouse\temp_dump\flower_test_img.pkl','wb')
    pickle.dump(file=fp,obj=test_img_dict)
    fp.close()
    print('Save test image complete!')
def YieldTrainImage(img_addr_dict=None):
    '''
    给定的图片地址字典（只对训练集操作）：扩展200倍，yield输出
    :param img_addr_dict:
    :return:yield输出每张训练图扩展200倍后的图片矩阵，标签矩阵
    '''
    if not img_addr_dict:
        #读取地址,假如没有地址字典传入，则读取存储的字典
        fp=open(r'D:\Data warehouse\temp_dump\flower_dict.pkl','rb')
        img_addr_dict=pickle.load(fp)
        fp.close()
    #因为字典key没有次序，为了保证标签有一致性，规定次序
    flower=['dandelion','tulips','roses','sunflowers','daisy']
    for index,fname in enumerate(flower):
        label_=[0]*5
        label_[index]=1
        y_=np.asarray(label_*200).reshape((-1,5))
        for addr_ in img_addr_dict[fname]['train']:
            #每张训练图片扩展200倍后，yield输出
            img_=imread(addr_)[:,:,:3]
            x_=da.AlexEnlarge(img_)
            yield (x_,y_)

#DataConver函数不能用，多个函数会使得yield无法正常使用
def DataConver(img_file_root=r'D:\Data warehouse\5 flower\flower_photos'):
    img_addr_dict=GetFlowerPicAddr(img_file_root)
    #SaveTestImage(img_addr_dict)
    YieldTrainImage(img_addr_dict)
