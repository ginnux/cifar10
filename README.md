# CIFAR-10 Classification CIFAR-10图像分类任务
A deep-learning program of computer vision, which finishes CIFAR-10 pictures classification.  

一个基于计算机视觉的深度学习项目，用于完成CIFAR-10数据集的分类任务。

## Necessary Python Lib 依赖库
torch (1.13 for cuda 11.7)  
torchvision  
matplotlib  
numpy  

## Necessary Environment 依赖环境
python 3.10  
NVIDIA cuda 11.7  

## Parameter 参数
### trainMode:
set 0 for checking testset  
set 1 for continue training  
set 2 for creating a new model to train  
set 3 for checking both testset and trainset  
### REPEAT:
times of training  

## Notice 注意
The first time to train maybe you can set trainMode 2 and REPEAT 20.  
Training with CPU will be VERY VERY SLOW!  

首次训练可以将trainmode设置为2，REPEAT设置为20。
如果基于CPU训练会非常缓慢，建议使用NVIDIA显卡训练。
