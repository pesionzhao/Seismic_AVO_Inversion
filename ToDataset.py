"""
@File    : ToDataset.py
@Author  : Pesion
@Date    : 2023/12/29
@Desc    : 
"""
import scipy.io as scio
from util.utils import read_yaml
import numpy as np
import argparse
from util.utils import read_mat, augdata, img2line, back_model, v2seis, generate_dataset, extract_seis, update_back
import matplotlib.pyplot as plt


def create():
    """
    Notes:
        由原始数据生成训练集与测试集, 这里使用npy文件, 指定原始文件目录, 指定下采样倍率, 默认
    """
    vp = np.load(r'F:\Graduate\First\Repo\Reader\red_sea_vp9441_2801.npy')
    vs = np.load(r'F:\Graduate\First\Repo\Reader\red_sea_vs.npy')
    rho = np.load(r'F:\Graduate\First\Repo\Reader\red_sea_rho.npy')
    vp, vs, rho = extract_seis(vp.T, vs.T, rho.T, 0.4, 0.1)  # 下采样到原数据的[layer_ratio, trace_ratio]倍
    vp_back, vs_back, rho_back = back_model(vp, vs, rho)
    generate_dataset(vp_back, vs_back, rho_back, cutoff_f=1)


def update():  # 用不同截止频率更新低频模型
    trainmat = scio.loadmat('Red_train_dataset.mat')
    testmat = scio.loadmat('Red_test_dataset.mat')
    update_back(trainmat, testmat, f=1)

def check():  # 自定义查看mat数据并显示
    back_mat = read_mat('Red_test_dataset.mat', ['vp_back', 'vs_back', 'rho_back','vp_aug'])
    label_mat = read_mat('Red_label.mat', ['vp', 'vs', 'rho'])
    traces = label_mat['vp'].shape[1]
    user_input = None
    plt.show()
    while user_input != 'q':
        user_input = input(f'请输入查看道数(0-{traces})（按q键结束）: ')
        for key, value in label_mat.items():
            plt.figure()
            plt.plot(value[:, int(user_input)])
            back_data = back_mat[key + '_back']
            plt.plot(back_data[:, int(user_input)])
            plt.title(key)
        plt.show()


if __name__ == '__main__':
    create()
    # check()