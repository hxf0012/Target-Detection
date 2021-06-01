import numpy as np
# from utils.config import opt
# from model import FasterRCNNVGG16
# from trainer import FasterRCNNTrainer
# from data.util import  read_image
# from utils.vis_tool import vis_bbox
# from utils import array_tool as at
# import torch as t
# import scipy.io as sio
# import os
# from os.path import isfile, join
# import cv2
'''比赛用到的分数计算'''
def CPSC2019_challenge(ECG,index):
    '''
    This function is your detection function, the input parameters is self-definedself. # 这个函数是你的检测函数，输入参数是自定义的。

    INPUT:
    ECG: single ecg data for 10 senonds # 10秒单次心电图数据分析
    .
    .
    .

    OUTPUT:
    hr: heart rate calculated based on the ecg data # 根据心电图数据计算心率
    qrs: R peak location detected beased on the ecg data and your algorithm # 基于ecg数据的r峰定位及其算法
    '''

    #批量
    # center_path = '/result/0924/'
    #单张
    center_path='/VOCdevkit/VOC2007-ecg/ecg_data/qrs_1/qrs/'
    index = (int(index) - 1) * 5 # (1601-1)*5=8000
    print(index)
    print(type(index))

    qrs = []
    for i in range(5):  # 外层循环开始
        index2 = index
        print(i)
        index2 = index2 + i
        index2 = str(index2)
        print(index2)
        lines = open(center_path + '00' + index2 + '_c.txt').readlines()  # 打开文件，读入每一行

        for s in lines:
            temp = s.split('\n')[0]
            temp = float(temp)
            # filter=temp-1000*i
            if ((temp >= 0.5 * 500) & (temp <= 9.5 * 500)):
                qrs.append(temp)

    print(qrs)

    qrs.sort(reverse=False)
    # 心率计算60/(p-p间隔)即60/(小格数*0.04)
    hr = np.array([loc for loc in qrs if (loc > 5.5 * 500 and loc < 5000 - 0.5 * 500)])
    print(hr)
    hr = round(60 * 500 / np.mean(np.diff(hr)))  # diff函数就是执行的是后一个元素减去前一个元素。np.mean求均值
    print(hr)
    # print(qrs)

    return hr, qrs































#
#
#
# import numpy as np
# from utils.config import opt
# from model import FasterRCNNVGG16
# from trainer import FasterRCNNTrainer
# from data.util import  read_image
# from utils.vis_tool import vis_bbox
# from utils import array_tool as at
# import torch as t
# import scipy.io as sio
# import os
# from os.path import isfile, join
# import cv2
#
# def CPSC2019_challenge(ECG,index):
#     '''
#     This function is your detection function, the input parameters is self-definedself. # 这个函数是你的检测函数，输入参数是自定义的。
#
#     INPUT:
#     ECG: single ecg data for 10 senonds # 10秒单次心电图数据分析
#     .
#     .
#     .
#
#     OUTPUT:
#     hr: heart rate calculated based on the ecg data # 根据心电图数据计算心率
#     qrs: R peak location detected beased on the ecg data and your algorithm # 基于ecg数据的r峰定位及其算法
#     '''
#     print(index)
#     index = str((int(index) - 1)*5)
#     # 单张
#     # datapath2 = '/VOCdevkit/VOC2007-ecg/test_image/one_test/'
#     # 批量
#     datapath2 = '/VOCdevkit/VOC2007-ecg/test_image/JPEGImages/'
#     img_name2 ='00'+ index + '.jpg'
#     jpg_dirtory = os.path.join(datapath2, img_name2)
#
#     img = read_image(jpg_dirtory)  # 矩阵
#     img = t.from_numpy(img)[None]
#
#     faster_rcnn = FasterRCNNVGG16()
#     trainer = FasterRCNNTrainer(faster_rcnn).cuda()
#     # 加载模型
#     trainer.load(
#         '/checkpoints/fasterrcnn_08191723_0.8634294227743174.pth')
#     opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model
#     _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
#
#     # print(_bboxes[0])
#     # image = cv2.imread(jpg_dirtory)  # 图片
#     bboxs = at.tonumpy(_bboxes[0])
#     name = at.tonumpy(_labels[0]).reshape(-1)
#     # print(name)
#
#     qrs1 = []
#     # qrs
#     for i in range(len(name)):
#         xmin = int(round(float(bboxs[i, 1])))
#         xmax = int(round(float(bboxs[i, 3])))
#         x_c = int((xmax - xmin) / 2) + xmin
#         x_c = float(x_c * 5)
#         if ((x_c >= 0.5 * 500) & (x_c <= 9.5 * 500)):         # 去头尾
#             qrs1.append(x_c)
#         # print(qrs1)
#     qrs = qrs1
#     # hr = 10
#     # qrs = np.arange(1, 5000, 500)
#
#     qrs.sort(reverse=False) #排序
#
#     # 心率计算60/(p-p间隔)即60/(小格数*0.04)
#     hr = np.array([loc for loc in qrs if (loc > 5.5 * 500 and loc < 5000 - 0.5 * 500)])
#     print(hr)
#     hr = round(60 * 500 / np.mean(np.diff(hr)))  # diff函数就是执行的是后一个元素减去前一个元素。np.mean求均值
#     print(hr)
#     print(qrs)
#
#
#     return hr, qrs
#
#
#
#
#
#
