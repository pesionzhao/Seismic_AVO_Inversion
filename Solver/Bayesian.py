"""
@File    : Bayesian.py
@Author  : Pesion
@Date    : 2023/10/28
@Desc    : 
"""
import numpy as np

from Solver import Solver


class Bayesian(Solver.Solver):
    def __init__(self, cfg, log_vp, log_vs, log_rho, layers):
        self.name = 'Linear_Bayesian-Solver'
        log_vp, log_vs, log_rho = np.log(log_vp), np.log(log_vs), np.log(log_rho)
        if len(log_vp.shape) == 1:
            self.vp_mean = np.mean(log_vp)
            self.vs_mean = np.mean(log_vs)
            self.rho_mean = np.mean(log_rho)
            self.cov = np.cov([log_vp, log_vs, log_rho])
        elif len(log_vp.shape) == 2:
            # # version 1 每一道有每一道的均值和方差,想stack起来之后用,但这个多的一维不知道怎么用
            # vp_mean = []
            # vs_mean = []
            # rho_mean = []
            # cov = []
            # for i in range(log_vp.shape[0]):
            #     ivp_mean = np.mean(log_vp[i])
            #     ivs_mean = np.mean(log_vs[i])
            #     irho_mean = np.mean(log_rho[i])
            #     icov = np.cov(np.stack([log_vp[i],log_vs[i],log_rho[i]]))
            #     vp_mean.append(ivp_mean)
            #     vs_mean.append(ivs_mean)
            #     rho_mean.append(irho_mean)
            #     cov.append(icov)
            # self.vp_mean = np.array(vp_mean)
            # self.vs_mean = np.array(vs_mean)
            # self.rho_mean = np.array(rho_mean)
            # self.cov = np.array(cov)
            self.vp_mean = np.mean(log_vp)
            self.vs_mean = np.mean(log_vs)
            self.rho_mean = np.mean(log_rho)
            self.cov = np.cov(np.stack([log_vp.flatten(), log_vs.flatten(), log_rho.flatten()]))
        else:
            raise ValueError('vp_los dim > 2, I do not know what does the dimension mean')
        self.m_mean = np.hstack([np.full((layers,), self.vp_mean),
                                 np.full((layers,), self.vs_mean),
                                 np.full((layers,), self.rho_mean)])
        # 由于在正演时将vpvsrho展成一维去做，所以这里的协方差矩阵要相应改变，由论文中写的[3*3]的矩阵变为[3*layers, 3*layers],
        # 原来的每个值广播为对角子矩阵
        self.m_cov = np.kron(self.cov, np.identity(layers))
        self.error = 0

    def __str__(self):
        return f'反演策略为{self.name}\nvp, vs, rho均值为[{self.vp_mean:3f}, {self.vs_mean:3f}, {self.rho_mean:3f}]'

    def one_step(self, pre, jacobian, residual, ntraces, IRLS=False, Reg=None): #　todo 更正参数，residual?IRLS?
        if ntraces == 1:
            miu_obs = jacobian @ self.m_mean.T[..., np.newaxis]
            sigma_obs = jacobian @ self.m_cov @ jacobian.T + self.error
            inv1 = (jacobian @ self.m_cov).T @ np.linalg.pinv(sigma_obs)
            miu_m_obs = self.m_mean.T + inv1 @ (residual.T.reshape(-1, ntraces) - miu_obs).squeeze()
            result = np.exp(miu_m_obs)
        else:
            self.m_mean = np.stack([self.m_mean for _ in range(ntraces)])
            # self.m_mean = np.log(pre).T
            miu_obs = jacobian @ self.m_mean[..., np.newaxis]
            sigma_obs = jacobian @ self.m_cov @ jacobian.transpose([0, 2, 1]) + self.error
            inv1 = (jacobian @ self.m_cov).transpose([0, 2, 1]) @ np.linalg.pinv(sigma_obs)
            miu_m_obs = self.m_mean + (
                        inv1 @ (residual.transpose([0, 2, 1]).reshape(ntraces, -1, 1) - miu_obs)).squeeze()
            result = np.exp(miu_m_obs).T

        return result
