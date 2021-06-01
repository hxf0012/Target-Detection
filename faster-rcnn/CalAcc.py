import cv2
import numpy as np
import matplotlib.pylab as plt
import scipy.misc as misc
import torch.nn as nn
import math
import pandas as pd
'''计算准确率'''
count=0
for i in range(290):
    img = cv2.imread('./result21/gen_image_0_{}.png'.format(i))
    image = np.array(img, np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    target=cv2.imread(('./result21/original_image_0_{}.png'.format(i)))
    binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 输出为三个参数
    cv2.drawContours(target, contours, -1, (0, 0, 255), 1)
    a = []
    b = []
    print('number:',i)
    for j in range(len(contours)):
        cnt = contours[j]#取第一条轮廓
        M = cv2.moments(cnt)#计算第一条轮廓的各阶矩,字典形式
        #这两行是计算中心点坐标
        if M['m00']==0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print("pre:")
        print(cx,cy)
        a.append(cx)
        a.append(cy)
        cv2.circle(target, (cx, cy), 1, 170, -1)#绘制中心点
    label = cv2.imread('./result21/label_image_0_{}.png'.format(i))
    image = np.array(label, np.uint8)
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(target, contours, -1, (255, 0,0), 1)
    for j in range(len(contours)):
        cnt= contours[j]#取第一条轮廓
        M = cv2.moments(cnt)#计算第一条轮廓的各阶矩,字典形式
        #这两行是计算中心点坐标
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print('label:')
        print(cx,cy)
        b.append(cx)
        b.append(cy)
        cv2.circle(target, (cx, cy), 1, 225, -1)#绘制中心点
    c=0
    for j in range(len(a)//2):
        n=j*2
        if c >= (len(b) // 2):
            break
        for m in range(len(b)//2):
            m=m*2
            x=a[n]-b[m]
            y=a[n+1]-b[m+1]
            loss=math.sqrt(x**2+y**2)
            if loss <=10:
                count=count+1
                c=c+1
                print(loss)
                print('count:', count)


    # plt.imshow(target)
    # plt.show()
print('Accuracy Rate:',count/436)


cv2.waitKey(0)



