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

### Option 3 yoloV4 使用说明

> 1、克隆darknet

git clone https://github.com/AlexeyAB/darknet
> 2、编译项目

cpu环境下:

cd darknet
make

gpu环境下:

(1)修改darknet的Makefile文件，GPU/CUDNN/CUDNN_HALF/=1；OPENCV=0在服务器下可不用
(2)然后执行make clean和make命令，重新进行编译

> 3、下载预训练权重文件

可以打开链接https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights下载好之后，放在darknet的文件夹下

> 4、测试

./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights data/dog.jpg


./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4-tiny.conv.29 data/dog.jpg

> 测试结果

在目录darknet下的predictions.jpg是产生的预测结果图像文件