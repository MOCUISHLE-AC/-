# -*- coding: utf-8 -*-
import os
#用getcwd()获取当前目录
filedir = "F:\data\data_aishell\wav\wav"
#获取目录列表
filenames=os.listdir(filedir)
#以写的方式打开一个文本
f=open('sph_flist.txt','w+')
for root, dirs, files in os.walk(filedir):
    for file in files:
        path = os.path.join(root,file)
        #print(path)
        #把dir写到1.txt文本里面
        f.writelines(path+'\n')
f.close()