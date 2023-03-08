import numpy as np
from matplotlib import pyplot as plt

def plt_loss(loss_list):
    epoch_list = list(range(len(loss_list)))
    plt.figure(1)
    plt.title('Loss Curve')  # 图片标题
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('Loss')  # y轴变量名称
    plt.plot(epoch_list, loss_list, 'bp-', label=u"联邦学习损失曲线")
    plt.xticks(epoch_list)
    plt.legend()  # 画出曲线图标
    plt.show()  # 画出图像


def plt_acc(acc_list):
    epoch_list = list(range(len(acc_list)))
    plt.figure(1)
    plt.title('ACC Curve')  # 图片标题
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('ACC')  # y轴变量名称
    plt.plot(epoch_list, acc_list, 'bp-', label=u"联邦学习准确率曲线")
    plt.xticks(epoch_list)
    plt.legend()  # 画出曲线图标
    plt.show()  # 画出图像

