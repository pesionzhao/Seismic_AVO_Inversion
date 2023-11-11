"""
@File :test02.py 
@Author :Pesion
@Date :2023/9/20
@Desc : 
"""
import scipy.io as scio
from util.utils import read_yaml
import numpy as np
import argparse
from util.utils import read_mat
import matplotlib.pyplot as plt
#
#
# a = scio.loadmat('dataG.mat')
# b = scio.loadmat('dataGo.mat')
# x = read_yaml('NetWork/config/Network.yaml')
# opt = argparse.Namespace(**x)
# c = a-b
#
# print(0)

# read_mat('./NetWork/dataset/Marmousi_1_3401.mat')
read_mat('./NetWork/Gan/WGan_Data.mat')
# read_mat('./NetWork/dataset/DataTest02_27.mat')
# read_mat('./NetWork/output/result.mat', True)
plt.show()