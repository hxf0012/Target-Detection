import os
import xml.etree.ElementTree as ET  # ET类是专门用来解析标注xml文件的
import numpy as np
from .util import read_image

############voc_dataset将xml文件和图片文件加载到一起




#读取voc数据集 实现魔术方法__getitem__（以便pytorch的DataLoader读取数据集）
# 返回其中一张的图片img:numpy矩阵,标签label: 0-19,标注框box:（[ymin,xmin,ymax,xmax],是否难以标注difficult: 0 or 1
# 一张图片可以有多个box(R,4)和label(R,)和difficult(R,) 那么将返回多维numpy数组形式
#如果要训练自己的数据集 那么请修改这里的魔术方法读取自己的数据集

class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False, ):
# 参数datadir由utils.config的voc_data_dir而来，是数据集存放的地址,trainval是main文件里
        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split)) # main文件里存放图片名字，训练测试用到的图片名字存在不同txt里
        self.ids = [id_.strip() for id_ in open(id_list_file)]  #类表解析：读取上面的txt 按行读取后装入一个列表
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES  # 在最下面，是voc数据集所有物体name的tuple


    def __len__(self):
        return len(self.ids) # 数据集的数量 就是ids列表的长度

    def get_example(self, i):  # 魔术方法：从数据集列表ids中 选取一个进行xml解析
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]      # 列表中选取一个数据
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))   # 找到名字对应的xml 用ET进行解析
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):    # 找到所有标注的object
            # 当没有启用difficult 但是我们找到了标注的difficult物体（标注值为1） 跳过这个object
            # .text方法是获取xml标签里的内容 比如obj.find('difficult')=<difficult>1<difficult/>
            # obj.find('difficult').text = 1 （string类型数据 要转成int）
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')      #  找到标注框boundbox
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])    # 列表解析：[ymin,xmin,ymax,xmax] 减一是为了让像素的索引从0开始
            name = obj.find('name').text.lower().strip()  # name标注的对应VOC_BBOX_LABEL_NAMES中的一个
            label.append(VOC_BBOX_LABEL_NAMES.index(name))  # label就是VOC_BBOX_LABEL_NAME中name的索引 范围0-19
        bbox = np.stack(bbox).astype(np.float32)   # 将box从list转成np.float32类型
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # 由于pytorch不支持np.bool 我们要将difficult 转成np.bool后再转成unint8

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)           # 调用data/util 中的read_image方法 读取图片数据

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'qrs',
    # 'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
