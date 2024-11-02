"""
@File :demo.py
@Author :Pesion
@Date :2023/9/12
@Desc : 
"""
import argparse
import random

import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np

from Forward.Zoeppritz import Zoeppritz
from Forward.Zoeppritz_Complex import ZoeppritzComp
from Forward.Aki_Richards import Aki_Richards
from Forward.Simplify_Aki_Richards import Simplify_Aki_Richards
from util.DataSet import Train_DataSet, Read_Marmousi, Set_origin, Read_Marmousi2
from Builder import SimulateBuilder, Bayesian_builder
from Solver.Bayesian import Bayesian
from Solver.Solver import LM_solver, GN_solver, SB_solver, GD_solver
from Regularization.Regularization import IRLS, NoReg, User_L2
from util.utils import read_yaml, plot_single_trace, plot_multi_traces, plot_result
import time

solver_map = {'LM': LM_solver, 'GN': GN_solver, 'SB': SB_solver, 'GD': GD_solver}
reg_map = {'IRLS': IRLS, 'NoReg': NoReg, 'User_L2': User_L2}
cfg = read_yaml('config/prestack.yaml')
cfg = argparse.Namespace(**cfg)
# 数据准备部分
# datapath = 'DataSet/Marmousi_dataset_dt=8ms.mat'
datapath = cfg.datapath
datareader = Read_Marmousi2()
datareader.read(datapath)
print(datareader)

# settings
dt0 = cfg.dt0
wave_f = cfg.wave_f
wave_n0 = cfg.wave_n0
if isinstance(cfg.use_trace, list):
    use_trace = slice(cfg.use_trace[0], cfg.use_trace[1], cfg.use_trace[2])
else:
    use_trace = cfg.use_trace

if isinstance(cfg.use_layer, list):
    use_layer = slice(cfg.use_layer[0], cfg.use_layer[1], cfg.use_layer[2])
else:
    use_layer = cfg.use_layer
# trace = 1
# theta = [5, 10, 15, 20, 25, 30, 35, 40]
# theta = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]  # 用于和matlab代码做对比
theta = cfg.theta
config_path = './config/MyCfg.yaml'
build_method = 'GD'

inv_cfg = read_yaml(config_path, build_method)

dataset = Train_DataSet(datareader.vp, datareader.vs, datareader.rho, theta, use_trace, use_layer)
print(dataset)
settings = Set_origin(dataset.layers, dt0, wave_f, wave_n0, theta)
settings.setup()
print(settings)


# dataset.show()

# 反演流程
def run():
    # 三种正演模型的选择
    # forwardmodel = Aki_Richards(dataset.ntraces, dataset.layers)  # TODO layer-1的处理
    # forwardmodel = Simplify_Aki_Richard(dataset.ntraces, dataset.layers)  # TODO layer-1的处理
    forwardmodel = ZoeppritzComp(dataset.ntraces, dataset.layers)  # TODO layer-1的处理

    # 正演结果显示
    # forwardmodel.forward(dataset.vp, dataset.vs, dataset.rho, dataset.theta_rad, settings.wavemat)
    # forwardmodel.showresult(dt0, 1, settings.theta)

    # 定义solver
    # solver = LM_solver(config_path=config_path, method=build_method)
    solver_cfg = inv_cfg['solver']
    solver = solver_map[solver_cfg['name']](solver_cfg)

    # 定义Regularization
    # Reg = NoReg()
    reg_cfg = inv_cfg['reg']
    reg = reg_map[reg_cfg['name']]()

    # 初始化正则化
    reg.set_up(reg_cfg, layers=dataset.layers, ntraces=dataset.ntraces)

    # 实例化工厂
    worker = SimulateBuilder(dataset, settings, forwardmodel, solver, reg)
    # 反演
    pre, loss = worker.inverseRun()
    # 显示反演结果
    plot_result(pre, dataset.vp, dataset.vs, dataset.rho,
                dataset.vp_back, dataset.vs_back, dataset.rho_back)
    # 显示损失曲线
    plt.figure()
    plt.plot(loss)
    plt.title('loss curve')
    plt.show()

# 正演耗时计算
def cal_time():
    forwardmodel = Zoeppritz(dataset.ntraces, dataset.layers)
    t0 = time.time()
    forwardmodel.forward(dataset.vp, dataset.vs, dataset.rho, dataset.theta_rad, settings.wavemat)
    t1 = time.time()
    # forwardmodel.showresult(dt0, 0, settings.theta)
    print('Zoeppritz正演用时%f' % (t1 - t0))

    forwardmodel1 = Aki_Richards(dataset.ntraces, dataset.layers)  # TODO layer-1的处理
    t2 = time.time()
    forwardmodel1.forward(dataset.vp, dataset.vs, dataset.rho, dataset.theta_rad, settings.wavemat)
    t3 = time.time()
    # forwardmodel1.showresult(dt0, 0, settings.theta)

    print('Aki_Richards正演用时%f' %(t3-t2))
    # # scio.savemat('aki_new.mat', {"data":forwardmodel1.cal_data})
    # plt.show()

# 贝叶斯测试
def beyasian_test():
    vsvp = dataset.vs / dataset.vp  # 应该为先验
    generate_obs = ZoeppritzComp(dataset.ntraces, dataset.layers)
    obs = generate_obs.forward(dataset.vp, dataset.vs, dataset.rho, dataset.theta_rad, settings.wavemat)
    forwardmodel = Simplify_Aki_Richards(dataset.ntraces, dataset.layers, vsvp=vsvp, theta=dataset.theta_rad,
                                         wavmtx=settings.wavemat)
    random_log_index1 = random.randint(1, 100)
    random_log_index2 = random.randint(1, 100)
    log_vp = np.vstack([datareader.vp[:, random_log_index1], datareader.vp[:, random_log_index2]])
    log_vs = np.vstack([datareader.vs[:, random_log_index1], datareader.vs[:, random_log_index2]])
    log_rho = np.vstack([datareader.rho[:, random_log_index1], datareader.rho[:, random_log_index2]])

    # obsaki = forwardmodel.forward(dataset.vp, dataset.vs, dataset.rho, dataset.theta_rad, settings.wavemat)
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # ax1.imshow(obs[0])
    # ax1.set_title('obs')
    # ax2.imshow(obsaki[0])
    solver = Bayesian(cfg, dataset.vp, dataset.vs, dataset.rho, dataset.layers)
    worker = Bayesian_builder(dataset, settings, forwardmodel, solver, reg=NoReg())
    pre = worker.inverseRun()
    # plot_single_trace(pre, dataset.vp, dataset.vs, dataset.rho, solver.m_mean)
    plot_multi_traces(pre, dataset.vp, dataset.vs, dataset.rho)
    print('done')


if __name__ == '__main__':
    run()
