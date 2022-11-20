# cifar10
A deep-learning program of computer vision, which finishes CIFAR-10 pictures classification.  

## Necessary Python Lib
torch (1.13 for cuda 11.7)  
torchvision  
matplotlib  
numpy  

## Necessary Environment
python 3.10  
NVIDIA cuda 11.7  

## Parameter
### trainMode:
set 0 for checking testset  
set 1 for continue training  
set 2 for creating a new model to train  
set 3 for checking both testset and trainset  
### REPEAT:
times of training  

## Notice
The first time to train maybe you can set trainMode 2 and REPEAT 20.  
Training with CPU will be VERY VERY SLOW!  
