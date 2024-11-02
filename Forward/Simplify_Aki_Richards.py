"""
@File :Simplify_Aki_Richards.py 
@Author :Pesion
@Date :2023/9/17
@Desc : 
"""
from Forward.BasicForward import ForwardModel
import numpy as np
from numpy import sin, cos, arcsin, arccos
from scipy.linalg import block_diag
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool


class Simplify_Aki_Richards(ForwardModel):
    def __init__(self, ntrace, nt0, vsvp, theta, wavmtx):
        self.ntraces = ntrace  # TODO 加统一的数据预处理用于抛出异常
        self.layers = nt0
        self.createOp(vsvp, theta, nt0, wavmtx)

    def __str__(self):
        return '正演模型为线性简化Aki-Richards'

    def createOp(self, vsvp, theta, nt0, wav_mtx):
        """

        Args:
            vsvp: vs/vp
            theta: 入射角
            nt0: 层数
            wav_mtx: [layers, layers]

        Returns:
            np.array([G1, G2, G3]): aki-richards方程的系数
            self.op: Gm=d中的G [ntraces, layers*ntheta, 3*layers]

        Note:
            由于这里的m为三参数展平化处理后的，在进行构建正演算子时必须构建为对角阵，这会使得增加layer^2倍的计算内存, cpu算会很慢，gpu会爆显存\n
            解决办法初步设想 a) for循环 b)分块计算 c)更改正演策略使其不用对角阵

        """
        # theta = np.deg2rad(theta)

        theta = theta[:, np.newaxis] if vsvp.size > 1 else theta
        vsvp = vsvp[:, np.newaxis].T if vsvp.size > 1 else vsvp

        G1 = 1.0 / (2.0 * cos(theta) ** 2) + 0 * vsvp
        G2 = -4.0 * vsvp ** 2 * np.sin(theta) ** 2
        G3 = 0.5 - 2.0 * vsvp ** 2 * sin(theta) ** 2
        # G = np.array([G1, G2, G3])

        if self.ntraces == 1:
            ntheta = len(theta)
            # G= np.hstack([Gi for Gi in G])
            D = np.diag(np.ones(nt0 - 1), k=1) - np.diag(np.ones(nt0), k=0)  # 差分矩阵
            D[-1] = 0
            D = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
            G = np.vstack([np.hstack([np.diag(G1[itheta] * np.ones(nt0)), np.diag(G2[itheta] * np.ones(nt0)),
                                      np.diag(G3[itheta] * np.ones(nt0))])
                           for itheta in range(ntheta)])  # 生成可以通过矩阵乘法进行卷积的矩阵

            self.W = block_diag(*[wav_mtx for _ in range(ntheta)])  # 子波矩阵, * 表示遍历
            self.G = G @ D  # G in "WGm = d"
            self.D = D
            self.op = self.W @ self.G  # 得到"Ax = b"中的A
        else:
            ntheta = len(theta)
            D = np.diag(np.ones(nt0 - 1), k=1) - np.diag(np.ones(nt0), k=0)
            D[-1] = 0
            D = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
            # Garray = []
            # for i in range(self.ntraces):
            #     Gi = np.vstack([np.hstack([np.diag(G1[i, itheta]),np.diag(G2[i, itheta]),np.diag(G3[i, itheta])])
            #                     for itheta in range(ntheta)])
            #     Garray.append(Gi)
            G = np.zeros((self.ntraces, ntheta * nt0, 3 * nt0))
            for i in range(self.ntraces):
                for j in range(ntheta):
                    G[i, j * nt0:(j + 1) * nt0, 0:nt0] = np.diag(G1[i, j])
                    G[i, j * nt0:(j + 1) * nt0, nt0:2 * nt0] = np.diag(G2[i, j])
                    G[i, j * nt0:(j + 1) * nt0, 2 * nt0:3 * nt0] = np.diag(G3[i, j])

            self.W = block_diag(*[wav_mtx for i in range(ntheta)])  # * 表示遍历
            self.G = G@D
            self.D = D
            self.op = self.W@self.G
            # self.G = np.stack([G_ @ D for G_ in G])
            # self.op = np.stack([self.W @ G_ for G_ in self.G])

        return

    def create_sparse_Op(self, vsvp, theta, nt0, wav_mtx):
        """

        Args:
            vsvp: vs/vp
            theta: 入射角
            nt0: 层数
            wav_mtx: [layers, layers]

        Returns:
            np.array([G1, G2, G3]): aki-richards方程的系数
            self.op: Gm=d中的G

        Note:
            利用稀疏矩阵解决了createOp爆内存的问题

        """
        # theta = np.deg2rad(theta)

        theta = theta[:, np.newaxis] if vsvp.size > 1 else theta
        vsvp = vsvp[:, np.newaxis].T if vsvp.size > 1 else vsvp

        G1 = 1.0 / (2.0 * cos(theta) ** 2) + 0 * vsvp
        G2 = -4.0 * vsvp ** 2 * np.sin(theta) ** 2
        G3 = 0.5 - 2.0 * vsvp ** 2 * sin(theta) ** 2
        # G = np.array([G1, G2, G3])

        if self.ntraces == 1:
            ntheta = len(theta)
            # G= np.hstack([Gi for Gi in G])
            # D = np.diag(np.ones(nt0 - 1), k=1) - np.diag(np.ones(nt0), k=0)  # 差分矩阵
            # D[-1] = 0
            # D = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
            D = sp.diags([np.append(-np.ones(nt0-1),0), np.ones(nt0)], [0,1])# 差分稀疏矩阵
            D = sp.block_diag([D for _ in range(3)])
            G = sp.vstack([sp.hstack([sp.diags(G1[itheta]), sp.diags(G2[itheta]),
                                      sp.diags(G3[itheta])])
                           for itheta in range(ntheta)])  # 生成可以通过矩阵乘法进行卷积的矩阵

            self.W = sp.block_diag([wav_mtx for _ in range(ntheta)])  # 子波矩阵, * 表示遍历
            self.G = G @ D  # G in "WGm = d"
            self.D = D
            self.op = self.W @ self.G  # 得到"Ax = b"中的A
        else:
            self.op = []
            for i in range(self.ntraces):
                ntheta = len(theta)
                # G= np.hstack([Gi for Gi in G])
                # D = np.diag(np.ones(nt0 - 1), k=1) - np.diag(np.ones(nt0), k=0)  # 差分矩阵
                # D[-1] = 0
                # D = block_diag(*[D for _ in range(3)])  # 用列表批量化diag
                D = sp.diags([np.append(-np.ones(nt0-1),0), np.ones(nt0)], [0,1])# 差分稀疏矩阵
                D = sp.block_diag([D for _ in range(3)])
                G = sp.vstack([sp.hstack([sp.diags(G1[i,itheta]), sp.diags(G2[i,itheta]),
                                          sp.diags(G3[i,itheta])])
                               for itheta in range(ntheta)])  # 生成可以通过矩阵乘法进行卷积的矩阵

                self.W = sp.block_diag([wav_mtx for _ in range(ntheta)])  # 子波矩阵, * 表示遍历
                self.G = G @ D  # G in "WGm = d"
                self.D = D
                self.op.append(self.W @ self.G)

        return

    def forward_run(self, m, theta, show=False, t0=None):  # TODO m写的有点累赘，之后可以优化一下
        """

        Args:
            wav_mtx: 子波矩阵
            theta: 入射角
            nt0: 采样点 layers
            show: 是否显示合成后的地震数据
            t0: 采样时间

        Returns:
            cal_data: 正演得到的地震数据

        """
        # 单道正演
        if self.ntraces == 1:
            m = m.T.ravel()  # 注意是按行展平化，所以要先进行转置
            ntheta = len(theta)
            self.obs_data = np.dot(self.op, m)  # 计算"Ax"得到合成地震数据/仿真观测数据
            self.Rpp = self.G @ m  # 反射系数
            cal_data = self.obs_data.reshape(ntheta, -1).T  # 同样要注意rehsape的次序，所以要先reshpe再转置而不能直接reshape成指定维度
            if show:
                plt.figure()
                plt.imshow(cal_data, cmap="gray", extent=(theta[0], theta[-1], t0[-1], t0[0]))
                plt.axis("tight")
                plt.xlabel('incident angle')
                plt.ylabel('time')
                plt.title('seismic data from Aki-Ri')

        # 多道正演 仅仅比单道正演多了一维，需要stack
        else:
            m = np.vstack([m[..., i].T.ravel() for i in range(self.ntraces)])
            ntheta = len(theta)
            self.obs_data = np.stack([np.dot(self.op[i], m[i]) for i in range(self.ntraces)])
            # self.Rpp = np.dot(np.dot(G, D), m)
            cal_data = np.stack([data.reshape(ntheta, -1).T for data in
                                 self.obs_data])  # 同样要注意rehsape的次序，所以要先reshpe再转置而不能直接reshape成指定维度
            if show:  # 只显示一个剖面
                plt.figure()
                plt.imshow(cal_data[1], cmap="gray", extent=(theta[0], theta[-1], t0[-1], t0[0]),
                           vmin=-np.abs(cal_data).max(), vmax=np.abs(cal_data).max())
                plt.axis("tight")
                plt.xlabel('incident angle')
                plt.ylabel('time')
                plt.title('seismic data from Aki-Ri')

        return cal_data

    def forward_sparse_run(self, m, theta, show=False, t0=None):  # TODO m写的有点累赘，之后可以优化一下
        """

        Args:
            wav_mtx: 子波矩阵
            theta: 入射角
            nt0: 采样点 layers
            show: 是否显示合成后的地震数据
            t0: 采样时间

        Returns:
            cal_data: 正演得到的地震数据

        """
        # 单道正演
        if self.ntraces == 1:
            m = m.T.ravel()  # 注意是按行展平化，所以要先进行转置
            ntheta = len(theta)
            self.obs_data = self.op@m # 计算"Ax"得到合成地震数据/仿真观测数据
            self.Rpp = self.G @ m  # 反射系数
            cal_data = self.obs_data.reshape(ntheta, -1).T  # 同样要注意rehsape的次序，所以要先reshpe再转置而不能直接reshape成指定维度
            if show:
                plt.figure()
                plt.imshow(cal_data, cmap="gray", extent=(theta[0], theta[-1], t0[-1], t0[0]))
                plt.axis("tight")
                plt.xlabel('incident angle')
                plt.ylabel('time')
                plt.title('seismic data from Aki-Ri')

        # 多道正演 仅仅比单道正演多了一维，需要stack
        else:
            m = np.vstack([m[..., i].T.ravel() for i in range(self.ntraces)])
            ntheta = len(theta)
            self.obs_data = np.stack([self.op[i]@m[i] for i in range(self.ntraces)])
            # self.Rpp = np.dot(np.dot(G, D), m)
            cal_data = np.stack([data.reshape(ntheta, -1).T for data in
                                 self.obs_data])  # 同样要注意rehsape的次序，所以要先reshpe再转置而不能直接reshape成指定维度
            if show:  # 只显示一个剖面
                plt.figure()
                plt.imshow(cal_data[1], cmap="gray", extent=(theta[0], theta[-1], t0[-1], t0[0]),
                           vmin=-np.abs(cal_data).max(), vmax=np.abs(cal_data).max())
                plt.axis("tight")
                plt.xlabel('incident angle')
                plt.ylabel('time')
                plt.title('seismic data from Aki-Ri')

        return cal_data

    def forward(self, vp, vs, rho, theta, wavemat):
        m = np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)
        self.cal_data = self.forward_run(m, theta, show=False, t0=None)
        # self.cal_data = self.forward_sparse_run(m, theta, show=False, t0=None)
        return self.cal_data

    def jacobian(self, vp=None, vs=None, rho=None, theta=None, wav_mtx=None):
        return self.op

    def showresult(self, dt0, trace, theta):
        t = np.arange(self.layers) * dt0
        plt.figure()
        if self.ntraces != 1:
            plt.imshow(self.cal_data[trace], cmap="gray", extent=(theta[0], theta[-1], t[-1], t[0]))
            plt.title('synthetic data from Aki-Ri-Nonliner, trace = ' + str(trace))
        else:
            plt.imshow(self.cal_data, cmap="gray", extent=(theta[0], theta[-1], t[-1], t[0]))
            plt.title('synthetic data from Aki-Ri-Nonliner')
        plt.axis("tight")
        plt.xlabel('incident angle')
        plt.ylabel('time')
