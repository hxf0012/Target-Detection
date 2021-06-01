# Target-Detection

### Option 1 mmdetection使用说明

>  数据存放路径

/mmdetection/data
> 环境切换

conda activate 虚拟环境名
>  运行代码

python tools/train.py configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --gpus 1 --work-dir r1027
(python tools/train.py r1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --gpus 1)

> 测试代码

python tools/test.py r1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py r1027/epoch_12.pth --show-dir results/1027
> 保存结果

python demo/image_demo.py r1027/cascade_rcnn_x101_64x4d_fpn_1x_coco.py r1027/epoch_12.pth 

### Option 2 FasterRcnn 使用说明

> 数据

VOCdevkit/VOC2007 

> 训练

train.py

> 测试

单张测试test_one.py | 批量测试test_images.py