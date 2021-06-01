import torch
import matplotlib.pyplot as plt
import os
import numpy as np

path = 'C:\Software\pycharm\simple-faster-rcnn-pytorch-master-chenyun\loss_result\\'
save_path = 'C:\Software\pycharm\simple-faster-rcnn-pytorch-master-chenyun\loss_result\\'
x = torch.linspace(0,10,10)
print(type(x))

# mAp画图
mAp_data = np.loadtxt(path + 'mAp.txt').flatten()
mAp_data = torch.tensor(mAp_data)
print(type(mAp_data))
print(mAp_data)
plt.plot(x.data.numpy(),mAp_data.data.numpy(),'b-',lw = 2)
plt.legend(['mAp'],loc = 'upper left')
plt.savefig(save_path + 'mAp' +'.jpg')
plt.show()

#loss画图
loss_data = open(path + 'loss.txt')
loss = torch.zeros(10,5)
k = 0
for i in range(10):
    data = loss_data.readline()
    print(data)
    print(type(data))
    lens = len(data)
    j = 0
    for i in range(lens):
        if data[i] == ':' :
            loss[k,j] = float(data[i+2:i+8])
            j = j + 1
    k = k+1
print(loss)
label = ['rpn_loc_loss','rpn_cls_loss','roi_loc_loss','roi_cls_loss','total_loss']
for i in range(5):
    plt.plot(x.data.numpy(),loss[:,i].data.numpy(),'b-',lw = 2)
    plt.legend([label[i]],loc = 'upper right')
    plt.savefig(save_path + label[i] +'.jpg')
    plt.show()















