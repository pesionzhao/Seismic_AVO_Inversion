"""
@File :Regularization.py 
@Author :Pesion
@Date :2023/9/18
@Desc : 
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.linalg import block_diag
from util.utils import soft_thresh

class Regularization(metaclass=ABCMeta):
    @abstractmethod
    def set_up(self, cfg, layers, ntraces):
        """

        Args:
            cfg: 配置文件mat,用于定义正则化超参数
            layers:
            ntraces:
        Notes:
            根据配置文件初始化正则化项

        """
        pass

    @abstractmethod
    def update(self, pre, residual, jacobian):
        """

        Args:
            pre: 输入
            residual: 残差
            jacobian: 雅可比

        Notes:
            例如IRLS/Split Bergman的正则化项与输入有关,所以在每次迭代时需要根据当前计算正则化项,得到新的Ax=b中的A和b

        Returns:
            添加正则化项后的jacobian, 添加正则化项后的residual

        """
        pass


class NoReg(Regularization):
    def __str__(self):
        return 'Regularization: None'
    def set_up(self, cfg, layers, ntraces):
        pass

    def update(self, pre, residual, jacobian):
        return residual, jacobian


class IRLS(Regularization):
    def __str__(self):
        return f'Regularization: IRLS, alpha={self.alpha}, eps={self.eps}'
    def set_up(self, cfg, layers, ntraces):
        D = np.diag(np.ones(layers - 1), k=1) - np.diag(np.ones(layers), k=0)  # 差分矩阵
        D[-1] = 0
        self.L = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
        self.alpha = cfg["alpha"]
        self.eps = cfg["eps"]
        self.ntraces = ntraces

    def update(self, pre, residual, jacobian):
        w = self.L @ pre  # 纵向不连续性
        np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
        w = np.where(np.abs(w) > self.eps, 1 / np.abs(w), 1 / self.eps)
        W = np.diag(w) if self.ntraces == 1 else np.stack([np.diag(w[:, i]) for i in range(self.ntraces)])
        RegOp = np.sqrt(self.alpha / 2) * np.sqrt(W) @ self.L
        jacobian = np.vstack((jacobian, RegOp)) if self.ntraces == 1 else np.hstack(
            (jacobian, RegOp))
        residual = np.hstack((residual, np.zeros((RegOp.shape[-1])))) if self.ntraces == 1 else \
            np.hstack((residual, np.zeros((self.ntraces, RegOp.shape[-1]))))
        return residual, jacobian


class Tikhonov(Regularization):
    pass


class User_L2(Regularization):
    def __str__(self):
        return 'custom L2 Regularization(Specific for SB)'
    def set_up(self, cfg, layers, ntraces):
        self.layers = layers
        D = np.diag(np.ones(layers - 1), k=1) - np.diag(np.ones(layers), k=0)  # 差分矩阵
        D[-1] = 0
        self.L = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
        self.ntraces = ntraces
        self.d_ = np.zeros(3 * (self.layers)) if self.ntraces == 1 \
            else np.zeros((3 * (self.layers), self.ntraces))
        self.b_ = np.zeros(3 * (self.layers)) if self.ntraces == 1 \
            else np.zeros((3 * (self.layers), self.ntraces))
        self.alpha = cfg['alpha']
        self.lamb = cfg['lamb']
        self.tau = cfg['tau']

    def step(self, pre):
        self.RegOp = np.sqrt(self.lamb / 2) * self.L
        alpha = 0.0001
        cof = self.alpha / self.lamb
        self.Regdata = (np.sqrt(self.lamb / 2) * (self.d_ - self.b_)).T

        # subquestion2
        dataregs2 = (self.RegOp @ pre) + self.b_
        # self.d, sub2loss = self.Ista_run(self.d,np.eye(self.Reg.shape[0]),dataregs2,50,None) # TODO 为什么ISTA不行
        self.d_ = soft_thresh(0.2, dataregs2)  # TODO lambda 应该怎么确定
        # d_, loss2 = self.Ista_run(d_,np.eye(len(d_)),dataregs2,20)
        loss2 = 0
        #
        if self.ntraces != 1:
            self.RegOp = np.stack([self.RegOp for _ in range(self.ntraces)])

    def update(self, pre, residual, jacobian):
        self.step(pre)
        jacobian = np.vstack((jacobian, self.RegOp)) if self.ntraces == 1 else np.hstack(
            (jacobian, self.RegOp))
        residual = np.hstack((residual, self.Regdata))
        self.b_ = self.b_ + self.tau * (self.RegOp[0] @ pre - self.d_)
        return residual, jacobian


class User_Reg(Regularization):
    def __str__(self):
        return 'custom L1 Regularization(Specific for SB)'
    def step(self, pre):
        self.RegOp = np.sqrt(self.lamb / 2) * self.L
        alpha = 0.0001
        cof = self.alpha / self.lamb
        self.Regdata = np.sqrt(self.lamb / 2) * (self.d_ - self.b_)

        # subquestion2
        dataregs2 = (self.RegOp @ pre) + self.b_
        # self.d, sub2loss = self.Ista_run(self.d,np.eye(self.Reg.shape[0]),dataregs2,50,None) # TODO 为什么ISTA不行
        self.d_ = soft_thresh(10, dataregs2)  # TODO lambda 应该怎么确定
        # d_, loss2 = self.Ista_run(d_,np.eye(len(d_)),dataregs2,20)
        loss2 = 0

        if self.ntraces != 1:
            self.RegOp = np.stack([self.RegOp for _ in range(self.ntraces)])

        return self.RegOp, self.Regdata.T, [self.b_, self.d_, loss2]

    def set_up(self, cfg, layers, traces):
        D = np.diag(np.ones(layers - 1), k=1) - np.diag(np.ones(layers), k=0)  # 差分矩阵
        D[-1] = 0
        self.L = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
        self.ntraces = traces
        self.d_ = np.zeros(3 * (self.layer + 1)) if self.ntraces == 1 \
            else np.zeros((3 * (self.layer + 1), self.ntraces))
        self.b_ = np.zeros(3 * (self.layer + 1)) if self.ntraces == 1 \
            else np.zeros((3 * (self.layer + 1), self.ntraces))
        self.alpha = cfg['TV']['alpha']
        self.lamb = cfg['TV']['lamb']
        self.tau = cfg['TV']['tau']

    def step(self, pre):
        self.RegOp = np.sqrt(self.lamb / 2) * self.L
        alpha = 0.0001
        cof = self.alpha / self.lamb
        self.Regdata = np.sqrt(self.lamb / 2) * (self.d_ - self.b_)

        # subquestion2
        dataregs2 = (self.RegOp @ pre) + self.b_
        # self.d, sub2loss = self.Ista_run(self.d,np.eye(self.Reg.shape[0]),dataregs2,50,None) # TODO 为什么ISTA不行
        self.d_ = self._soft_thresh(10, dataregs2)  # TODO lambda 应该怎么确定
        # d_, loss2 = self.Ista_run(d_,np.eye(len(d_)),dataregs2,20)
        loss2 = 0

        if self.ntraces != 1:
            self.RegOp = np.stack([self.RegOp for _ in range(self.ntraces)])

        return self.RegOp, self.Regdata.T, [self.b_, self.d_, loss2]

    def update(self, pre, residual, jacobian, ntraces):
        self.step(pre)
        jacobian = np.vstack((jacobian, self.RegOp)) if ntraces == 1 else np.hstack(
            (jacobian, self.RegOp))
        residual = np.hstack((residual, self.Regdata))
        self.b_ = self.b_ + self.tau * (self.RegOp @ pre - self.d_)
        return residual, jacobian
