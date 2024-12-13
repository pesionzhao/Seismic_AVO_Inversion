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
    def one_step(self, pre, jacobian, residual, ntraces, Reg=None):
        """

        Args:
            pre: 输入
            jacobian: 雅可比矩阵
            residual: 与观测数据的残差
            ntraces: 道数
            Reg: 正则化项

        Notes:

            对于求解Gm=d问题,正则化项要拼接在G和d后面形成新的Gm=d,所以格式为[Regop, Regdata], 所以反问题更新为

            [G  ]      [d      ]

            |   | m = |        |

            [Reg]     [Regdata]

        Returns:
            迭代输出

        """
        pass

    def step(self, iter):
        pass


class GD_solver(Solver):
    def __init__(self, cfg):
        self.name = 'GD-Solver'
        self.iters = cfg["iters"]
        self.step_size = cfg["step_size"]
        self.milestone = cfg["milestone"]
        self.status = 0

    def __str__(self):
        return f'反演策略为梯度下降\nstep_size={self.step_size}, milestone为{self.milestone}'

    def one_step(self, pre, jacobian, residual, ntraces, Reg=None):
        lam = 0.001
        # lam = 0.005  # TODO 如何设定lambda
        layers = int(pre.shape[-1]/3)
        if isinstance(self.step_size[self.status], list):
            vp_step = np.full(layers, self.step_size[self.status][0])
            vs_step = np.full(layers, self.step_size[self.status][1])
            rho_step = np.full(layers, self.step_size[self.status][2])
            self.step_size[self.status] = np.hstack([vp_step, vs_step, rho_step])
        if Reg is not None:
            RegOp, Regdata = Reg[0], Reg[-1]
            # 由于多道处理会多一个维度，所以拼接时的维度也会从第一维变为第二维
            jacobian = np.vstack((jacobian, RegOp)) if ntraces == 1 else np.hstack(
                (jacobian, RegOp))
            residual = np.hstack((residual, Regdata))

        if ntraces == 1:
            delta = - self.step_size[self.status] * (jacobian.T @ residual)
            pre = pre + delta
        else:
            for i in range(ntraces):
                pre[:, i] = pre[:, i] - self.step_size[self.status] * (jacobian[i].T @ residual[i])
        return pre

    def step(self, iter):
        if self.status==len(self.milestone):
            return
        if iter>self.milestone[self.status]:
            self.status+=1



class LM_solver(Solver):
    def __init__(self, cfg):
        self.name = 'LM-Solver'
        self.iters = cfg["iters"]
        self.lamb = cfg["lamb"]
        self.step_size = cfg["step_size"]

    def __str__(self):
        return '反演策略为LM\nlambda=%f, step_size=%f' % (self.lamb, self.step_size)

    def one_step(self, pre, jacobian, residual, ntraces, Reg=None):
        global alpha, eps, L
        lam = 0.001
        # lam = 0.005  # todo 如何设定lambda
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

    def one_step(self, pre, jacobian, residual, ntraces, Reg=None):
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

    def one_step(self, pre, jacobian, residual, ntraces, Reg=None):
        pre = self.sub1_solver.one_step(pre, jacobian, residual, ntraces)
        return pre
