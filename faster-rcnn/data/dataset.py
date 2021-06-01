from __future__ import absolute_import
from __future__ import division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):  # 去正则化（去归一化），img维度为[[B,G,R],H,W],因为caf预训练模型输入为BGR 0-255图片，pyt预训练模型采用RGB 0-1图片
    if opt.caffe_pretrain:     # 如果采用caf预训练模型，则返回 img[::-1, :, :]
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        # reshape成与img相同维度才可以相加，caf_normalize之前有减均值预处理，现在还原回去。
        return img[::-1, :, :]   # 将BGR转换为RGB图片（python [::-1]为逆序输出）
    # approximate un-normalize for visualize   可视化的近似非标准化
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255
    # 如果不采用，则返回(img * 0.225 + 0.45).clip(min=0, max=1) * 255
    # pyt_normalze中标准化为减均值除以标准差，现在乘以标准差加上均值还原回去，转换为0-255
    # clip作用：给定一个范围[min, max]，数组中值不在这个范围内的，会被限定为这个范围的边界


# 采用pytorch预训练模型对图片预处理（归一化），函数输入的img为0-1
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])     # 首先设置channel=（channel-mean）/std 在RGB三个维度上归一化的参数
    img = normalize(t.from_numpy(img))  # (ndarray) → Tensor之后归一化
    return img.numpy()


# 采用caffe预训练模型时对输入图像进行标准化（归一化），函数输入的img为0-1
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR，因为caffe预训练模型输入为BGR 0-255图片
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)  # reshape为与img维度相同
    img = (img - mean).astype(np.float32, copy=True)    # 减均值操作,并转换数据类型为float32型
    return img   # 返回img


# 函数输入的img为0-255，按照论文长边不超1000，短边不超600，按此比例缩放
def preprocess(img, min_size=600, max_size=1000):

    C, H, W = img.shape  # 读取图片格式：通道，高度，宽度
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    # 选小的比例，这样长和宽都能放缩到规定的尺寸（设置放缩比，选小的方便大的和小的都能够放缩到合适的位置）
    img = img / 255.  # 转换为0-1
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # resize到（H * scale, W * scale）大小，位于(min_size,max_size)之间，anti_aliasing为是否采用高斯滤波
    # 根据opt.caffe_pretrain是否存在，选择调用pytorch_normalze或者caffe_normalze对图像进行正则化
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        # 设置了图片的最小最大尺寸，本pytorch代码中min_size=600,max_size=1000
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data  # 从in_data中读取 img,bbox,label 图片
        _, H, W = img.shape  # 读取出图片的长和宽
        img = preprocess(img, self.min_size, self.max_size)  # 调用函数preprocess，按此比例缩放
        _, o_H, o_W = img.shape  # 读取放缩后图片的shape
        scale = o_H / H          # 放缩前后相除，得出放缩比因子
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))   # 按照缩放比因子重新调整bboxes框的大小，在util.py中有定义

        # horizontally flip
        img, params = util.random_flip(img, x_random=True, return_param=True)
        # 进行图片的随机反转，图片旋转不变性，增强网络的鲁棒性
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        # 同样地将bbox进行与对应图片同样的水平翻转，在util.py中有定义
        return img, bbox, label, scale


class Dataset:  # 训练集样本的生成
    def __init__(self, opt):
        self.opt = opt   # 初始化类
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        # 调用VOCBboxDataset中的get_example（）从数据集存储路径中将img, bbox, label, difficult 一个个的获取出来
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # 调用前面的Transform函数将图片,label进行最小值最大值放缩归一化，重新调整bboxes的大小，然后随机反转，最后将数据集返回
        return img.copy(), bbox.copy(), label.copy(), scale    # 将处理后的数据集返回
    def __len__(self):
        return len(self.db)


class TestDataset:  # 测试集样本的生成
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt   # 实例化类
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        # 调用缩放函数，因为是测试，不需要反转，没有bbox需要考虑，所以无需调用Transform
        #接调用preposses()函数进行最大值最小值裁剪然后归一化，就完成了测试数据集的处理
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
