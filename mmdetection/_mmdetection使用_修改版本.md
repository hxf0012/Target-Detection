# 1 .dat格式数据制作coco数据集

#### 步骤一 ：数据解析.dat     1024 * 64  double  转化    png或jpg   1024 * 64

运行文件 GenerateJPGForTrain.m

```matlab
clear;
close all;
dataLen=64*1024;
headLen=90;

flist=dir('./two/test2/*.dat');  % 路径
flist={flist.name};
jpgpath='./n_test2' % 这里的分号别加上
mkdir(jpgpath);
for i=1:length(flist)
    filename=flist{i};
    f = fopen(['./two/test2/' filename],'rb');  % 路径
    data=fread(f,headLen+dataLen,'double');
    data=data(headLen+1:headLen+dataLen);
    data=reshape(data,[1024,64]);
    data=data';
    data=mat2gray(data);
    imwrite(data,[jpgpath,'/',filename(1:end-4),'.jpg']);
    fclose(f);
end
```

####  步骤二 ：.txt  转化   .json

运行txttojson.py文件，会在制定文件夹下生成json文件

##### 需要修改处：txt文件所在路径及保存的json路径，共两处

```python
# -*- coding:utf-8 -*-
'''
用txt文件生成json，存放在和.png图片一起的文件夹png下
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
# 2500为txt总数
for i in range(2500):
    shapes = []
    #####需要修改处1####  此处为txt标注文件路径
    file = open('./bz_2/' + str(i+1) + '.txt')
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
        imagePath =str(i+1)+'.png'
        data = dict_json(shapes, imagePath)
        #####需要修改处2####  此处为生成json文件保存路径
        json_file = './png/'+str(i+1)+'.json'
        json.dump(data, open(json_file, 'w'))
```

#### 步骤三：png和json   一起转化成coco格式数据集

coco的目录不用自己新建，直接生成在运行目录下，需要修改的共5处

```python
'''
用.png图片和.json文件一起生成coco数据集
coco的目录不用自己新建，直接生成在运行目录下
'''
import os
import json
import numpy as np
import glob
import shutil
import cv2
from sklearn.model_selection import train_test_split

np.random.seed(41)

# 0为背景
classname_to_id = {
    "target": 1,

}

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        # from labelme import utils
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # h, w = img_x.shape[:-1]
        image['height'] = 64
        image['width'] = 1024
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".png") # 存图路径
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    ###修改处1：json和png图片一起在的文件夹
    labelme_path = "E:\\a_hexiaofang\project\py_project\\1_detection\data_process\png\\"
    
    ###修改处2：生成的coco数据集所在文件夹
    saved_coco_path = "./"
    
    print('reading...')
    
    # 创建文件
     ###修改处3：把image那一层去掉
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/images/train2017/" % saved_coco_path):
        os.makedirs("%scoco/images/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/images/val2017/" % saved_coco_path):
        os.makedirs("%scoco/images/val2017" % saved_coco_path)
        
    # 获取images目录下所有的joson文件列表
    print(labelme_path + "*.json")
    json_list_path = glob.glob(labelme_path + "*.json")
    print('json_list_path: ', len(json_list_path))
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=0.1, train_size=0.9)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
    for file in train_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/train2017/" % saved_coco_path)
        img_name = file.replace('json', 'png')
        
        ###修改处4：截取图像的名字，根据路径长度修改，此处的img_name_1应该为1.jpg
        img_name_1 = img_name[64:]  # 只要图片名字XX.png
        temp_img = cv2.imread(img_name)
        print("$$$$$$$$$train",img_name_1)
        try:
             ###修改处3：把image那一层去掉
            cv2.imwrite("{}coco/images/train2017/{}".format(saved_coco_path, img_name_1.replace('png', 'png')),temp_img)

        except Exception as e:
            print(e)
            print('Wrong Image:', img_name )
            continue
        # print(img_name + '-->', img_name.replace('png', 'png'))

    for file in val_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/val2017/" % saved_coco_path)
        img_name = file.replace('json', 'png')
         
        ###修改处5：截取图像的名字，根据路径长度修改，此处的img_name_1应该为1.jpg
        img_name_1 = img_name[64:]
        
        temp_img = cv2.imread(img_name)
        print("$$$$$$$$$val", img_name_1)
        try:
            ###修改处3：把image那一层去掉
            cv2.imwrite("{}coco/images/val2017/{}".format(saved_coco_path, img_name_1.replace('png', 'png')), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name)
            continue
        # print(img_name + '-->', img_name.replace('png', 'png'))

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)

```



## 2 mmdetection安装

教程链接 https://zhuanlan.zhihu.com/p/101202864

#### 步骤1：需要的环境配置（install.md)

```
安装pytorch、cuda、cudnn、anaconda、git
```

#### 步骤2：下载mmdetection，安装mmcv(补图)

```

```

#### 步骤3：训练自己的数据集，需修改处

**（1）使用coco数据集训练**

在mmdetection目录下新建data文件，最终数据文件目录mmdetection/data/coco/annotations,train,val三个文件夹

**（2）修改3个文件img_scale参数（考虑内存不足），共6处**

configs/_base_/datasets/ coco_detection.py    coco_instance_sematic.py   coco_instance.py三个py文件中的img_scale（600，400）

```Python
# coco_detection.py   coco_instance_sematic.py   coco_instance.py
data_root = 'data/coco/' # 可以修改
img_scale=(1024, 64) # 可以修改每个文件两处
```

**（3）修改model的基础网络py文件的num_classes参数**

configs/_base_/models/xxxxx.py（所有） 

```Python
# cascade每个文件有三处 
# faster每个文件有1处
# mask每个文件2处  
# retinanet 有1处 
# rpn的没有
# ssd300有1处
num_classes = 1（自己需要检测的类别个数，不需要计算背景）
```

**（4）修改为自己的类别名，共两处**

mmdet/core/evaluation/class_names.py    在coco classes处修改自己的类别名

```python
def coco_classes():
    return [
        'target'
    ]
```

mmdet/datasets/coco.py     在classes处修改自己的类别名

```python
CLASSES = ('target')
```

**（5）train**

tools功能类的api，创建自己的work-dir文件（不用操作，等训练的时候加上语句即可）



# 3 mmdetection训练与测试

**（1）训练：**

```python
# 第1次训练：运行代码
python tools/train.py configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --gpus 1（训练使用的gpu块数，一块） --work-dir hxf1027（自己工作文件的目录，可改）

# ctrl+c将进程中断，然后在work-dir中查看自己的log，json，faster_rnn_xxxxxx_coco.py 有三个文件。
# 保留faster_rnn_xxxxxx_coco.py文件，其他可以删除
对faster_rnn_xxxxxx_coco.py 文件中，可修改num_classes、img_scale、checkpoint_config、log_config中的参数。

# 第2次训练：运行代码
python tools/train.py hxf1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --gpus 1
```

**（6）（未用）查看训练的config_log参数**

python tools/analyze_logs.py plot_curve work-dir/xxxxxx.json --keys acc loss_cls loss_bbox loss_mask --out out.pdf 

 --keys 此处可以有很多json中的参数

--out 此处也可以是out.png  out.jpg，最终的保存下来的out.pdf结果可以在work-dir下找到

**（7）test**

python tools/test.py work-dir/faster_rnn_xxxxxx_coco.py work-dir/epoch_24.pth --show-dir xxxxx文件名

测试的图片从val中随机获取

```python
# 测试代码
python tools/test.py hxf1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py hxf1027/epoch_12.pth --show-dir hxf_results/1027（可改）
# 保存结果
#1 将test文件夹（里面是jpg图片）放在demo文件夹下，修改demo/image_demo_hxf.py文件的路径
#2 运行
python demo/image_demo_hxf.py hxf1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py hxf1027/epoch_12.pth 

```

**（8）（未用）关于测试图片的类别名命名为walk，seat，而不是id：002，003，040（之前（4）中定义类别名）**

demo/image_demo.py中的show_result_pyplot函数，找到mmdet/models/detectors/base.py文件中的328行，添加real_name = []  并将class_name = self.classes注释掉，改为class_name =real_name 

```python
  # class_names=self.CLASSES,
  改为class_name =real_name 
```

再次执行test代码，就可以发现测试图片上都是目前新定义的类别。



## 补充：具体使用

```python
# 数据存放路径
/mmdetection/data
# 环境切换
conda activate 虚拟环境名
# 运行代码
python tools/train.py configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --gpus 1 --work-dir hxf1027
(python tools/train.py hxf1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --gpus 1)
# 测试代码
python tools/test.py hxf1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py hxf1027/epoch_12.pth --show-dir hxf_results/1027
# 保存结果
python demo/image_demo_hxf.py hxf1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py hxf1027/epoch_12.pth 
```

