"""
@File :Builder.py 
@Author :Pesion
@Date :2023/9/13
@Desc : 
"""
from abc import ABCMeta, abstractmethod
from Forward.BasicForward import ForwardModel
from util.DataSet import Train_DataSet, Real_Dataset, Settings
from Solver.Solver import Solver
from Regularization.Regularization import Regularization
from tqdm import tqdm
import numpy as np


class Builder(metaclass=ABCMeta):
    @abstractmethod
    def calculate_residual(self, vp, vs, rho, obs):
        """计算损失"""
        pass

    @abstractmethod
    def inverseRun(self):
        """反演入口"""
        pass



class SimulateBuilder(Builder):
    def __init__(self, dataset: Train_DataSet, settings: Settings, forwardmodel: ForwardModel, solver: Solver,
                 reg: Regularization, obs=None):
        self.dataset = dataset
        self.settings = settings
        self.forwardmodel = forwardmodel
        self.reg = reg
        self.solver = solver
        if obs is None:
            self.obs = self.forwardmodel.forward(self.dataset.vp, self.dataset.vs, self.dataset.rho, self.dataset.theta_rad,
                                                 self.settings.wavemat)
        else:
            self.obs = obs
        # print(dataset)
        print(forwardmodel)
        print(solver)
        print(reg)

    def calculate_residual(self, vp, vs, rho, obs):
        cal_data = self.forwardmodel.forward(vp, vs, rho, self.dataset.theta_rad, self.settings.wavemat)
        res = np.squeeze((cal_data - obs).T.ravel()) if self.forwardmodel.ntraces == 1 \
            else (cal_data - obs).reshape(self.forwardmodel.ntraces, -1, order="F")  # 按列展开
        return res

    def solve(self, vp_back, vs_back, rho_back, obs):
        pre = np.vstack((vp_back, vs_back, rho_back)) if self.forwardmodel.ntraces != 1 else \
            np.stack((vp_back, vs_back, rho_back), axis=1).T.ravel()
        loss = []
        with tqdm(total=self.solver.iters) as t:
            for i in range(self.solver.iters):
                t.set_description(self.solver.name)
                res = self.calculate_residual(pre[:self.dataset.layers],
                                              pre[self.dataset.layers:2 * self.dataset.layers],
                                              pre[2 * self.dataset.layers:],
                                              obs)
                jacbian = self.forwardmodel.jacobian(pre[:self.dataset.layers],
                                                     pre[self.dataset.layers:2 * self.dataset.layers],
                                                     pre[2 * self.dataset.layers:],
                                                     self.dataset.theta_rad,
                                                     self.settings.wavemat)
                jac = self.forwardmodel.jacobian_(pre[:self.dataset.layers],
                                                pre[self.dataset.layers:2 * self.dataset.layers],
                                                pre[2 * self.dataset.layers:],
                                                self.dataset.theta_rad,
                                                self.settings.wavemat)
                res, jacbian = self.reg.update(pre, res, jacbian)
                pre = self.solver.one_step(pre, jacbian, res, self.dataset.ntraces)
                self.solver.step(i)
                rmse = np.linalg.norm(res)  # 显示的损失和回传的损失不一样哦
                t.set_postfix(loss=rmse)
                loss.append(rmse)
                t.update(1)
        return pre, loss

    def solve_sb(self, vp_back, vs_back, rho_back, obs):
        pre = np.vstack((vp_back, vs_back, rho_back)) if self.forwardmodel.ntraces != 1 else \
            np.stack((vp_back, vs_back, rho_back), axis=1).T.ravel()
        loss = []
        for i in range(self.solver.iters):
            with tqdm(total=self.solver.iters) as t:
                t.set_description(self.solver.name + ' (%d/%d)' % ((i + 1), self.solver.iters))
                for _ in range(self.solver.sub1_solver.iters):
                    res = self.calculate_residual(pre[:self.dataset.layers],
                                                  pre[self.dataset.layers:2 * self.dataset.layers],
                                                  pre[2 * self.dataset.layers:],
                                                  obs)
                    jacbian = self.forwardmodel.jacobian(pre[:self.dataset.layers],
                                                         pre[self.dataset.layers:2 * self.dataset.layers],
                                                         pre[2 * self.dataset.layers:],
                                                         self.dataset.theta_rad,
                                                         self.settings.wavemat)
                    res, jacbian = self.reg.update(pre, res, jacbian)
                    pre = self.solver.one_step(pre, jacbian, res, self.dataset.ntraces)
                    rmse = np.linalg.norm(res)  # 显示的损失和回传的损失不一样哦
                    t.set_postfix(loss=rmse)
                    t.update(1)
            loss.append(rmse)
        return pre

    def inverseRun(self):
        if self.solver.name == 'SB-solver':
            return self.solve_sb(self.dataset.vp_back, self.dataset.vs_back, self.dataset.rho_back, self.obs)
        else:
            return self.solve(self.dataset.vp_back, self.dataset.vs_back, self.dataset.rho_back, self.obs)


class Bayesian_builder(SimulateBuilder):
    def __int__(self, dataset,  settings, forwardmodel, solver, reg, obs):
        super(Bayesian_builder, self).__init__(dataset, settings, forwardmodel, solver, reg, obs)

    def solve(self, vp_back, vs_back, rho_back, obs):
        jacobian = self.forwardmodel.jacobian()
        pre = np.vstack((vp_back, vs_back, rho_back)) if self.forwardmodel.ntraces != 1 else \
            np.stack((vp_back, vs_back, rho_back), axis=1).T.ravel()
        pre = self.solver.one_step(pre=pre,jacobian=jacobian,residual=obs,ntraces=self.dataset.ntraces)
        return pre


