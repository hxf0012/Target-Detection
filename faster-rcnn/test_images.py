from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import torch as t
import scipy.io as sio
import os
from os.path import isfile, join
import cv2
import numpy as np

# 批量JPEGImages   单张one_test
# 加载图片
datapath = '/VOCdevkit/VOC2007-ecg/test_image/'  # 数据的路径
datapath2 = '/VOCdevkit/VOC2007-ecg/test_image/JPEGImages/'
for (root, dirs, files) in os.walk(join(datapath, 'JPEGImages')):  # 遍历路径下文件数据
    print(files)  # 输出files记录正在遍历的文件夹中的文件集合

data = files  # 将file的值赋给data变量
print(data)

# ref数据路径（原中心）
refpath = '/VOCdevkit/VOC2007-ecg/ecg_data/'  # 原中心数据的路径
save_path = '/result/1012/'
center_path = '/result/1012/'

ref_index = 0
i = 0
kk = 0
for i in range(2000):  # 外层循环开始

    img_name2 = data[i]
    print('图片名：', img_name2)
    txtname = str(img_name2)
    refname = int(int(txtname[1:6]) / 5 + 1)  # 例如8000除以5等于1600,加1对应1601
    refname = str(refname)
    print('refname：', refname)
    # print(txtname[0:6])

    # print(txtname[1:6])
    ref_index = int(int(txtname[1:6]) % 5)
    print('ref_index：', ref_index)

    dummy2 = sio.loadmat(join(refpath, 'ref', 'R_0' + refname))['R_peak'].squeeze()
    print(dummy2)

    jpg_dirtory = os.path.join(datapath2, img_name2)
    # print(jpg_dirtory)

    img = read_image(jpg_dirtory)
    # print(img)

    img = t.from_numpy(img)[None]

    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    # 加载模型
    trainer.load(
        '/checkpoints/fasterrcnn_09201838_0.8850640998285892.pth')
    opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)

    # print(_bboxes[0])
    image = cv2.imread(jpg_dirtory)
    bboxs = at.tonumpy(_bboxes[0])
    name = at.tonumpy(_labels[0]).reshape(-1)
    score = at.tonumpy(_scores[0]).reshape(-1)
    # print(name)

    center_t = open(center_path + txtname[0:6] + '_c.txt', 'w')
    # score_t = open(center_path + txtname[0:6] + '_s.txt', 'w')  # 目前不用写
    # 显示
    for i in range(len(name)):
        # print('文件数：',len(name))
        xmin = int(round(float(bboxs[i, 1])))
        ymin = int(round(float(bboxs[i, 0])))
        xmax = int(round(float(bboxs[i, 3])))
        ymax = int(round(float(bboxs[i, 2])))
        y_c = ymin + int(ymax / 2)

        x_c = int((xmax - xmin) / 2) + xmin  # 取中心
        # print('测试qrs位置：',x_c)

        temp_score = np.float(score[i])
        temp_score = '%.2f' % temp_score  # score保留小数点前两个
        # print('**********score********',temp_score)
        # print(type(temp_score))
        if 150 <= x_c <= 850:
            # 阈值
            if float(temp_score) >= 0.90:
                center_t.write(str(x_c + ref_index * 1000) + '\n')  # 记录中心
                # score_t.write(str(temp_score) + '\n')   # 目前不用写

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(temp_score), (xmin, 100), font, 1.2, (0, 255, 0), 2)  # 绿色文字
                # (255, 255, 255)白色
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (0, 255, 0), 2)  # 绿色框
                cv2.circle(image, (x_c, y_c), 10, (0, 0, 255), -1)  # 红色点检测点
        else:
            # 阈值
            if float(temp_score) >= 0.85:
                center_t.write(str(x_c + ref_index * 1000) + '\n')  # 记录中心
                # score_t.write(str(temp_score) + '\n')   # 目前不用写

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(temp_score), (xmin, 100), font, 1.2, (0, 255, 0), 2)  # 绿色文字
                # (255, 255, 255)白色
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (0, 255, 0), 2)  # 绿色框
                cv2.circle(image, (x_c, y_c), 10, (0, 0, 255), -1)  # 红色点检测点

    # k=0
    for j in range(len(dummy2)):
        # print('len(dummy2)=',len(dummy2))
        if ref_index * 1000 <= dummy2[j] < (ref_index + 1) * 1000:
            x_c_o = int(dummy2[j]) - ref_index * 1000  # 图片原中心
            print('图片名：', img_name2)
            # print('ref_index=',ref_index)
            # print('*******x_c_o=',x_c_o)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(x_c_o+ ref_index * 1000 ), (x_c_o - 30, 400), font, 1.2, (0, 255, 0), 2)  # 绿色文字
            cv2.circle(image, (x_c_o, 300), 10, (225, 225, 0), -1)  # 蓝色点原点

    # 保存
    cv2.imwrite(save_path + img_name2, image)

