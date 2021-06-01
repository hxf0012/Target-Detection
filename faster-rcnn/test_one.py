#%%

import os
import cv2
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import numpy as np
import os
from os.path import isfile, join
from torchvision.datasets import ImageFolder
'''单张测试'''
datapath = '/test_image/JPEGImages'

img = read_image('misc/001990.jpg')
img = t.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('/checkpoints/fasterrcnn_08152320_0.8695278719021664.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)

print(_bboxes[0])
image = cv2.imread('misc/001990.jpg')
bboxs = at.tonumpy(_bboxes[0])
name = at.tonumpy(_labels[0]).reshape(-1)
print(name)

for i in range(len(name)):
    xmin = int(round(float(bboxs[i, 1])))
    ymin = int(round(float(bboxs[i, 0])))
    xmax = int(round(float(bboxs[i, 3])))
    ymax = int(round(float(bboxs[i, 2])))
    cv2.rectangle(image, (xmin, ymin),
                  (xmax, ymax), (0, 255, 0), 2)

cv2.imwrite('/result/' + '001990'+ '.jpg', image)