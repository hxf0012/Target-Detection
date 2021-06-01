# -*- coding:utf-8 -*-

'''
仿照labelme的json文件写入自己的数据
用txt文件生成json，存放在和.png图片一起的文件夹下
'''
import cv2
import json
import numpy as np


def dict_json(shapes,imagePath):
    '''
    :param shapes: list
    :param imagePath: str
    :return: dict
    '''
    return {"shapes":shapes,'imagePath':imagePath}

def dict_shapes(points,label):
    return {'points':points,'label':label}

label='target'
for i in range(2500):
    shapes = []
    # 此处为txt标注文件路径
    file = open('C:\\n_bz\\' + str(i+1) + '.txt')
    for line in file.readlines():
        curLine=line.strip().split(" ")
        floatLine=list(map(int,curLine))#这里使用的是map函数直接把数据转化成为float类型
        # print(floatLine)
        floatLine[2] += floatLine[0]
        floatLine[3] += floatLine[1]
        floatLine = np.array(floatLine).reshape((2,2))
        floatLine  = floatLine.tolist()
        points = floatLine
        shapes.append(dict_shapes(points, label))
        imagePath =str(i+1)+'.jpg'
        data = dict_json(shapes, imagePath)
        json_file = './n_jpg/'+str(i+1)+'.json'
        json.dump(data, open(json_file, 'w'))

