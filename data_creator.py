#!C:\Python27\Python
# -*- coding=UTF-8 -*-

# y = kx + k2x2 + b

import numpy as np
size = 5000
#k=3.2
#b=4.3
k = 1.2
k2 = 5.4
b = 4.3

train = 'train_2.txt'
test = 'test_2.txt'
with open(train,'w') as f: #采用with构造，如果没有train文件，会自动创建，且结束后关闭文件
    for line in range (size):
        x = np.random.randn()#生成随机数符合标准正态分布
        x2 = np.random.randn()
        y = k * x + k2 * x2 + b        
        f.write("%f\t%f\t%f\n" %(x,x2,y))#写入文件
with open(test,'w') as f:
    for line in range (size):
        x = np.random.randn()
        x2 = np.random.randn()
        y = k * x + k2 * x2 + b        
        f.write("%f\t%f\t%f\n" %(x,x2,y))#写入文件