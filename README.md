# alexnet_transfer_learning
envirment:
python 3.5 ,tensorflow 1.2.0，pillow,matplotlib
How to:
make sure that you download the weight file(http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/),
the weight file and the code should be in the same file
/*********************************
确保下载了alexnet的权值文件，并把这个文件与代码放在一起（或者是自己在代码中声明路径）
**********************************/

file :
alexnet_tensorlow.ipynb -- implement of the alexnet mode      #alex_net模型的实现代码
class_name.py --to conver label to class name                 #输出的0~999标签转换成预测物体的名字
some jpeg image--test if the model work                       #测试用的图片
## Alexnet_tensorflow_extrator&trainer.ipynb
可以直接下载使用：alexnet提取图片特征或者是微调aelxnet以适应直接的数据集
## Alexnet_tensorflow_finetune.ipynb
Alexnet的基本实现代码
reference：
https://github.com/kratzert/finetune_alexnet_with_tensorflow
## Alexne_数据增强.ipynb
实现AlexNet里面用到的数据增强技术（第一种，也就是修改图片大小）
## 数据增强基本文件
1.data_augmentation.py ----数据增强通用的基本函数
2.FlowerData.py        ----针对实验用的5种花的数据的数据扩展函数，有参考意义
