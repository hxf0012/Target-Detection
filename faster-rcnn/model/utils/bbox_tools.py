import numpy as np
import numpy as xp

import six
from six import __init__


def loc2bbox(src_bbox, loc):   # 已知源预测bbox和位置偏差dx，dy，dh，dw，求目标框G
                               # loc预测回归
    if src_bbox.shape[0] == 0:
        # src_bbox：（R，4），R为bbox个数，4为左下角和右上角四个坐标(这里有误，按照标准坐标系中y轴向下，应该为左上和右下角坐标)
        return xp.zeros((0, 4), dtype=loc.dtype)   # 如果src_bbox.shape[0] == 0，也就是R==0，那么

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]      # 高ymax-ymin进行回归是要将数据格式从左上右下的坐标表示形式转化到中心点和长宽的表示形式
    src_width = src_bbox[:, 3] - src_bbox[:, 1]       # 宽xmax-xmin
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height     # y0+0.5h
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width      # x0+0.5w,计算出中心点坐标

    dy = loc[:, 0::4]     #  python [start:stop:step]，列表切片，隔3个取一个数据(4-1=3)
    dx = loc[:, 1::4]     #  把dx，dy，dh，dw的值单独提出来
    dh = loc[:, 2::4]     # 分别求出回归预测loc的四个参数来对源框bbox进行修正
    dw = loc[:, 3::4]
    #  RCNN中提出的边框回归：寻找原始proposal与近似目标框G之间的映射关系
    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    # np.newaxis的作用就是在这一位置增加一个一维，这样才可以计算
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]
    #  上面四行得到了回归后的目标框（Gx,Gy,Gh,Gw），包含中心点坐标和宽高
    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    #  loc.shape：（R，4），同src_bbox，生成一个和loc一样大小的全零矩阵
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w  #  由中心点转换为左上角和右下角坐标

    return dst_bbox   # 返回由源预测bbox和位置偏差dx，dy，dh，dw求解出的更接近目标检测GT的框


def bbox2loc(src_bbox, dst_bbox):  # 已知源框和目标框，求出其位置偏差,根据anchor来预测真实的目标的位置

    height = src_bbox[:, 2] - src_bbox[:, 0] + 1
    width = src_bbox[:, 3] - src_bbox[:, 1]  + 1
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width               # 计算源预测bbox中心点坐标和宽高

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0] + 1
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]  + 1
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width     # 计算目标bbox中心点坐标和宽高
    # 我们要考虑到除法的分母是不能为0的，而且式子中log内也不能为负数，不然会直接跳出显示错误
    eps = xp.finfo(height.dtype).eps   # 求出最小的正数（eps是取非负的最小值）
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)     # 将height,width与eps比较保证全部是非负

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)      # 按照公式求出偏移量,eps求出最小的正数

    loc = xp.vstack((dy, dx, dh, dw)).transpose()    # np.vstack按照行的顺序把数组给堆叠起来 然后转置
    return loc   #利用上述的公式求出偏移量的值tx,ty,tw,th完成了从bbox到loc的转化


def bbox_iou(bbox_a, bbox_b):    # 求两个bbox的相交的交并比

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:     # 确保bbox第二维为bbox的四个坐标（ymin，xmin，ymax，xmax）
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  #取两个iou左上最大值
    # tl为交叉部分框左上角坐标最大值
    # 为了利用numpy的广播性质，bbox_a[:, None, :2]的shape是(N,1,2)，bbox_b[:, :2]shape是(K,2)
    # 由numpy的广播性质，两个数组shape都变成(N,K,2)，也就是对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值

    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])    # 取两个iou右下最小值
    # br为交叉部分框右下角坐标最小值
    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 所有坐标轴上tl<br时，返回数组元素的乘积(y1max-yimin)X(x1max-x1min)，bboxa与bboxb相交区域的面积
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # 计算bboxa的面积
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    # 计算bboxb的面积
    return area_i / (area_a[:, None] + area_b - area_i)  #[:,None]把行转化为列
    # 计算并返回IOU的值


def __test():
    pass


if __name__ == '__main__':
    __test()

# 这个函数的作用就是产生(0,0)坐标开始的基础的9个anchor框
# 到底怎样进行目标检测？如何才能不漏下任何一个目标？那就是遍历的方法，不是遍历图片，而是遍历特征图
#原来anchor_scales=[8, 16, 32]  [64, 128, 256]
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    # 对特征图features以基准长度为16，选择合适的ratios和scales取基准锚点anchor_base，anchor的宽和高都是16。
    # 选择长度为16的原因是图片大小为600 * 800左右，基准长度16对应的原图区域是256 * 256
    # 考虑放缩后的大小有128 * 128，512 * 512比较合适
    # 根据基准点生成9个基本的anchor的功能，ratios=[0.5,1,2],  0.5：1 ， 1：2
    # anchor_scales=[8,16,32]是长宽比和缩放比
    # anchor_scales也就是在base_size的基础上再增加的量
    # 本代码中对应着三种面积的大小(16*8)^2 ,(16*16)^2  (16*32)^2  也就是128,256,512的平方大小
    py = base_size / 2.   # 三种面积乘以三种放缩比就刚刚好是9种anchor
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    # 按照长宽比和缩放比排列为9个anchor，#（9，4），注意：这里只是以特征图的左上角点为基准产生的9个anchor
    for i in six.moves.range(len(ratios)):
        # six.moves 是用来处理那些在python2 和 3里面函数的位置有变化的，直接用six.moves就可以屏蔽掉这些变化
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])  #生成9种不同比例的h和w
            # 其实这个函数一开始就只是以特征图的左上角为基准产生的9个anchor, 根本没有对全图的所有anchor的产生做任何的解释
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.   #计算出anchor_base画的9个框的左下角和右上角的4个anchor坐标值
    return anchor_base   # 该函数只是以特征图的左上角为基准产生的9个anchor,根本没有对全图的所有anchor的产生做任何的解释！
                         # 那所有的anchor是在 model / region_proposal_network里！！