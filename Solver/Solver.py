"""
@File :Solver.py 
@Author :Pesion
@Date :2023/9/12
@Desc : 
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Solver(metaclass=ABCMeta):
    @abstractmethod
    def one_step(self, pre, jacobian, residual, ntraces, IRLS=False, Reg=None):
        pass


class LM_solver(Solver):
    def __init__(self, cfg):
        self.name = 'LM-Solver'
        self.iters = cfg["iters"]
        self.lamb = cfg["lamb"]
        self.step_size = cfg["step_size"]

    def __str__(self):
        return '反演策略为LM\nlambda=%f, step_size=%f' % (self.lamb, self.step_size)

    def one_step(self, pre, jacobian, residual, ntraces, IRLS=False, Reg=None):
        global alpha, eps, L
        lam = 0.001
        # lam = 0.005  # todo 如何设定lambda
        if IRLS:
            RegOp = self._IRLSupdate(L, pre, alpha, eps)
            # 由于多道处理会多一个维度，所以拼接时的维度也会从第一维变为第二维
            jacobian = np.vstack((jacobian, RegOp)) if not self.model.multi else np.hstack(
                (jacobian, RegOp))
            residual = np.hstack((residual, np.zeros((RegOp.shape[-1])))) if not self.model.multi else \
                np.hstack((residual, np.zeros((self.model.traces, RegOp.shape[-1]))))

        if Reg is not None:
            RegOp, Regdata = Reg[0], Reg[-1]
            # 由于多道处理会多一个维度，所以拼接时的维度也会从第一维变为第二维
            jacobian = np.vstack((jacobian, RegOp)) if not self.model.multi else np.hstack(
                (jacobian, RegOp))
            residual = np.hstack((residual, Regdata))

        if ntraces == 1:
            delta = - self.step_size * np.linalg.pinv(jacobian.T @ jacobian + lam * np.identity(jacobian.shape[1])) @ (
                    jacobian.T @ residual)
            pre = pre + delta
        else:
            for i in range(ntraces):
                pre[:, i] = pre[:, i] - self.step_size * np.linalg.pinv(
                    jacobian[i].T @ jacobian[i] + lam * np.identity(jacobian[i].shape[1])) @ (
                                    jacobian[i].T @ residual[i])
        return pre


class GN_solver(Solver):
    def __init__(self, cfg):
        self.name = 'GN-Solver'
        self.iters = cfg["iters"]
        self.step_size = cfg["step_size"]

    def __str__(self):
        return '反演策略为GN\nstep_size=%f' % (self.step_size)

    def one_step(self, pre, jacobian, residual, ntraces, IRLS=False, Reg=None):
        if ntraces == 1:
            pre = pre - self.step_size * np.linalg.pinv(jacobian.T @ jacobian) @ (
                    jacobian.T @ residual)
        else:
            for i in range(ntraces):
                pre[:, i] = pre[:, i] - self.step_size * np.linalg.pinv(
                    jacobian[i].T @ jacobian[i]) @ (jacobian[i].T @ residual[i])
        return pre

class SB_solver(Solver):
    def __init__(self, cfg):
        solver_map = {'LM': LM_solver, 'GN': GN_solver}
        sub1_cfg = cfg['sub1']
        self.sub1_solver = solver_map[sub1_cfg['name']](sub1_cfg)
        self.iters = cfg['iters_outer']
        self.name = 'SB-solver'

    def __str__(self):
        return '反演策略为split-bregman\n子问题1迭代方法为%s' % self.sub1_solver.name
    def updata(self):
        pass
    def one_step(self, pre, jacobian, residual, ntraces, IRLS=False, Reg=None):
        pre = self.sub1_solver.one_step(pre, jacobian, residual, ntraces)
        return pre



