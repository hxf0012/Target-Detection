"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):   # 将数据转化为Numpy
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):  # 是pytorch的tensor
        return data.detach().cpu().numpy()  # 将变量从图中分离（使得数据独立，以后你再如何操作都不会对图，对模型产生影响），如果是gpu类型变成cpu的（cpu类型调用cpu方法没有影响），再转化为numpy数组



def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()