"""
@File :Aki_Richards.py 
@Author :Pesion
@Date :2023/9/17
@Desc : 
"""
from Forward.BasicForward import ForwardModel
import numpy as np
from numpy import sin, cos, arcsin, arccos
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


class Aki_Richards(ForwardModel):
    def __init__(self, ntraces, layers):
        self.obs_data = None
        self.ntraces = ntraces  # 0代表单道，多道即为道数
        if self.ntraces > 1:
            self.multi = True
        else:
            self.multi = False
        self.layers = layers

    def __str__(self):
        return '正演模型为Aki-Richards'

    def forward_single_layer(self, vp1, vp2, vs1, vs2, rho1, rho2, Theta_r):
        """
        Args:
            Theta_r: 入射角
        Returns:
            Rpp: 单层反射系数
        Notes:
            通过单层上下界面参数，计算单层反射系数

        """
        if self.multi:
            vp1 = vp1[:, np.newaxis]
            vp2 = vp2[:, np.newaxis]
            vs1 = vs1[:, np.newaxis]
            vs2 = vs2[:, np.newaxis]
            rho1 = rho1[:, np.newaxis]
            rho2 = rho2[:, np.newaxis]

        # 计算平均与残差
        vp = (vp1 + vp2) / 2
        vs = (vs1 + vs2) / 2
        rho = (rho1 + rho2) / 2

        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1

        # 计算平均角
        Const = np.sin(Theta_r) / vp1  # np中的三角函数必须为弧度制！
        temp = Const * vp2
        temp = np.where(temp > 1, 1, temp)
        temp = np.where(temp < -1, -1, temp)
        Theta_t = np.arcsin(temp)
        theta = (Theta_r + Theta_t) / 2

        # Aki-Richards方程的系数
        G1 = (0.5 * ((1 / np.cos(theta)) ** 2))
        G2 = -(4 * (vs / vp * np.sin(theta)) ** 2)
        G3 = 0.5 * (1 - 4 * (vs / vp * np.sin(theta)) ** 2)

        G = np.stack([G1, G2, G3], 2) if self.multi else np.stack([G1, G2, G3], 1)

        # m = np.array([dvp / vp, dvs / vs, drho / rho]).squeeze()
        # Rpp = np.stack([G[i, :, :] @ m[:, i] for i in range(G.shape[0])]) if self.multi else G @ m # old method
        if self.ntraces == 1:
            m = np.array([dvp / vp, dvs / vs, drho / rho]).squeeze()
            Rpp = G @ m
        else:
            m = np.stack([dvp / vp, dvs / vs, drho / rho], axis=1)
            Rpp = (G @ m).squeeze()

        return Rpp

    def forward_multi(self, vp, vs, rho, wav_mtx, theta):
        """

        Args:
            vp: 纵波序列
            vs: 横波序列
            rho: 密度序列
            wav_mtx: 子波矩阵
            theta: 入射角集

        Returns:
            Rpp: 整道反射系数

        Notes:
            目前只做模型，如果处理实际数据在没有真实的vpvsrho的情况下很多初始化没有进行，需要更改

        """
        Rpp = []
        for i in range(self.layers - 1):
            a1 = vp[i]
            a2 = vp[i + 1]
            b1 = vs[i]
            b2 = vs[i + 1]
            rho1 = rho[i]
            rho2 = rho[i + 1]
            iRpp = self.forward_single_layer(a1, a2, b1, b2, rho1, rho2, theta)
            Rpp.append(iRpp)

        Rpp = np.array(Rpp)
        Rpp = np.vstack([Rpp, np.zeros([1, Rpp.shape[-2], Rpp.shape[-1]])]) if self.multi else \
            np.vstack([Rpp, np.zeros(Rpp.shape[-1])])  # 添加全零行，行数补到和采样点数一样
        self.cal_data = np.stack(
            [wav_mtx @ Rpp[:, i, :] for i in range(Rpp.shape[-2])]) if self.multi else wav_mtx @ Rpp
        return self.cal_data

    def forward_multi_trace(self, vp, vs, rho, wav_mtx, theta):
        """

        Args:
            vp: 纵波序列
            vs: 横波序列
            rho: 密度序列
            wav_mtx: 子波矩阵
            theta: 入射角集

        Returns:
            Rpp: 整道反射系数

        Notes:
            完成多道同时计算正演，但内部还会有一个循环计算每一层的数据，下一个函数forward_multi_tracesAndlayers()可不使用循环直接通过矩阵计算

        """
        Rpp = np.zeros((self.ntraces, self.layers, len(theta)))
        for i in range(self.layers - 1):
            a1 = vp[i]
            a2 = vp[i + 1]
            b1 = vs[i]
            b2 = vs[i + 1]
            rho1 = rho[i]
            rho2 = rho[i + 1]
            iRpp = self.forward_single_layer(a1, a2, b1, b2, rho1, rho2, theta)
            Rpp[:, i, :] = iRpp

        wav_mtx = np.stack([*[wav_mtx for _ in range(self.ntraces)]])
        self.cal_data = wav_mtx @ Rpp
        return self.cal_data

    def forward_multi_tracesAndlayers(self, vp, vs, rho, wav_mtx, Theta_r):
        """
            Args:
                vp: [layer, trace]
                vs: [layer, trace]
                rho: [layer, trace]
                Theta_r: 入射角
            Returns:
                cal_data: 合成地震数据
            Notes:
                通过矩阵计算完成同时进行所有层所有道的正演计算，减小了以往通过for循环计算单层单道的时间
        """

        vp1 = vp[:-1]
        vp2 = vp[1:]
        vs1 = vs[:-1]
        vs2 = vs[1:]
        rho1 = rho[:-1]
        rho2 = rho[1:]

        if self.multi:
            vp1 = vp1[..., np.newaxis]
            vp2 = vp2[..., np.newaxis]
            vs1 = vs1[..., np.newaxis]
            vs2 = vs2[..., np.newaxis]
            rho1 = rho1[..., np.newaxis]
            rho2 = rho2[..., np.newaxis]

        # 计算平均与残差
        vp = (vp1 + vp2) / 2
        vs = (vs1 + vs2) / 2
        rho = (rho1 + rho2) / 2

        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1

        # 计算平均角
        Const = np.sin(Theta_r) / vp1  # np中的三角函数必须为弧度制！
        temp = Const * vp2
        temp = np.where(temp > 1, 1, temp)
        temp = np.where(temp < -1, -1, temp)
        Theta_t = np.arcsin(temp)
        theta = (Theta_r + Theta_t) / 2

        # Aki-Richards方程的系数
        G1 = (0.5 * ((1 / np.cos(theta)) ** 2))
        G2 = -(4 * (vs / vp * np.sin(theta)) ** 2)
        G3 = 0.5 * (1 - 4 * (vs / vp * np.sin(theta)) ** 2)

        G = np.stack([G1, G2, G3], -1) if self.multi else np.stack([G1, G2, G3], 1)

        # m = np.array([dvp / vp, dvs / vs, drho / rho]).squeeze()
        # Rpp = np.stack([G[i, :, :] @ m[:, i] for i in range(G.shape[0])]) if self.multi else G @ m # old method
        if self.ntraces == 1:
            m = np.array([dvp / vp, dvs / vs, drho / rho]).squeeze()
            Rpp = G @ m
        else:
            m = np.stack([dvp / vp, dvs / vs, drho / rho], axis=-2)
            Rpp = (G @ m).squeeze()

        Rpp = np.vstack([Rpp, np.zeros((1, Rpp.shape[1], Rpp.shape[2]))])  # 添加全零行
        Rpp = np.transpose(Rpp, [1, 0, 2])
        wav_mtx = np.stack([*[wav_mtx for _ in range(self.ntraces)]])
        self.cal_data = wav_mtx @ Rpp

        return self.cal_data

    def forward(self, vp, vs, rho, theta, wavemat):
        # self.multi = False
        # cal_data = np.zeros((self.ntraces,self.layers,len(theta)))
        # for i in range(self.ntraces):
        #     cal_data[i, :, :] = self.forward_multi(vp[:,i], vs[:,i], rho[:,i], wavemat, theta)
        # return cal_data
        if self.ntraces == 1:
            return self.forward_multi(vp, vs, rho, wavemat, theta)
        else:
            return self.forward_multi_tracesAndlayers(vp, vs, rho, wavemat, theta)
            # return self.forward_multi_trace(vp, vs, rho, wavemat, theta)

    def jacobian(self, vp, vs, rho, theta, wav_mtx):
        vp_nabla = []
        vs_nabla = []
        rho_nabla = []
        for i in range(self.layers - 1):
            vp_u = vp[i]
            vp_d = vp[i + 1]
            vs_u = vs[i]
            vs_d = vs[i + 1]
            den_u = rho[i]
            den_d = rho[i + 1]

            if self.multi:
                vp_u = vp_u[:, np.newaxis]
                vp_d = vp_d[:, np.newaxis]
                vs_u = vs_u[:, np.newaxis]
                vs_d = vs_d[:, np.newaxis]
                den_u = den_u[:, np.newaxis]
                den_d = den_d[:, np.newaxis]

            # 对于vp的偏导数
            temp1 = 8 * np.sin(theta) ** 2 * (den_d - den_u) * (vs_u + vs_d) ** 2
            temp1 = temp1 / ((den_u + den_d) * (vp_u + vp_d) ** 3)
            temp2 = 16 * np.sin(theta) ** 2 * (vs_d ** 2 - vs_u ** 2)
            temp2 = temp2 / (vp_u + vp_d) ** 3
            temp3 = (2 * vp_d) / ((vp_u + vp_d) ** 2 * np.cos(theta) ** 2)
            Der_r_vp_u = temp1 + temp2 - temp3
            vp_nabla.append(Der_r_vp_u)

            # 对于vs的偏导数
            temp1 = -8 * np.sin(theta) ** 2 * (den_d - den_u) * (vs_u + vs_d)
            temp1 = temp1 / ((den_u + den_d) * (vp_u + vp_d) ** 2)
            temp2 = 16 * np.sin(theta) ** 2 * vs_u
            temp2 = temp2 / (vp_u + vp_d) ** 2
            Der_r_vs_u = temp1 + temp2
            vs_nabla.append(Der_r_vs_u)

            # 对于rho的偏导数
            temp1 = -2 * den_d / (den_u + den_d) ** 2
            temp2 = 8 * np.sin(theta) ** 2 * den_d * (vs_u + vs_d) ** 2
            temp2 = temp2 / ((den_u + den_d) ** 2 * (vp_u + vp_d) ** 2)
            Der_r_den_u = temp1 + temp2
            rho_nabla.append(Der_r_den_u)

        if not self.multi:
            vp_nabla = np.vstack((np.array(vp_nabla), np.zeros(len(theta)))).T
            vs_nabla = np.vstack((np.array(vs_nabla), np.zeros(len(theta)))).T
            rho_nabla = np.vstack((np.array(rho_nabla), np.zeros(len(theta)))).T

            jac1 = np.vstack([np.diag(ivp) for ivp in vp_nabla])
            jac2 = np.vstack([np.diag(ivs) for ivs in vs_nabla])
            jac3 = np.vstack([np.diag(ir) for ir in rho_nabla])
            jac = np.hstack((jac1, jac2, jac3))
            G_mat = np.zeros(jac.shape)
            for n in range(jac.shape[0]):
                for m in range(3):
                    G_mat[n, (m) * 100: (m + 1) * 100] = (wav_mtx @ jac[n, m * 100: (m + 1) * 100].T).T
            W = block_diag(*[wav_mtx for _ in range(len(theta))])
            jac = W @ jac

        else:
            vp_nabla = np.vstack((np.array(vp_nabla), np.zeros((1, self.ntraces, len(theta)))))
            vs_nabla = np.vstack((np.array(vs_nabla), np.zeros((1, self.ntraces, len(theta)))))
            rho_nabla = np.vstack((np.array(rho_nabla), np.zeros((1, self.ntraces, len(theta)))))

            jac1 = np.stack([np.vstack([np.diag(vp_nabla[:, i, j]) for j in range(len(theta))])
                             for i in range(self.ntraces)])
            jac2 = np.stack([np.vstack([np.diag(vs_nabla[:, i, j]) for j in range(len(theta))])
                             for i in range(self.ntraces)])
            jac3 = np.stack([np.vstack([np.diag(rho_nabla[:, i, j]) for j in range(len(theta))])
                             for i in range(self.ntraces)])
            jac = np.concatenate((jac1, jac2, jac3), axis=2)
            # G_mat = np.zeros(jac.shape)
            # for n in range(jac.shape[0]):
            #     for m in range(3):
            #         G_mat[n, (m) * 100: (m + 1) * 100] = (wav_mtx @ jac[n, m * 100: (m + 1) * 100].T).T
            W = block_diag(*[wav_mtx for _ in range(len(theta))])
            jac = W @ jac

        return jac

    def jacobian_multilayers(self, vp, vs, rho, theta, wav_mtx):
        vp_nabla = []
        vs_nabla = []
        rho_nabla = []
        for i in range(self.layers - 1):
            vp_u = vp[i]
            vp_d = vp[i + 1]
            vs_u = vs[i]
            vs_d = vs[i + 1]
            den_u = rho[i]
            den_d = rho[i + 1]

            if self.multi:
                vp_u = vp_u[:, np.newaxis]
                vp_d = vp_d[:, np.newaxis]
                vs_u = vs_u[:, np.newaxis]
                vs_d = vs_d[:, np.newaxis]
                den_u = den_u[:, np.newaxis]
                den_d = den_d[:, np.newaxis]

            # 对于vp的偏导数
            temp1 = 8 * np.sin(theta) ** 2 * (den_d - den_u) * (vs_u + vs_d) ** 2
            temp1 = temp1 / ((den_u + den_d) * (vp_u + vp_d) ** 3)
            temp2 = 16 * np.sin(theta) ** 2 * (vs_d ** 2 - vs_u ** 2)
            temp2 = temp2 / (vp_u + vp_d) ** 3
            temp3 = (2 * vp_d) / ((vp_u + vp_d) ** 2 * np.cos(theta) ** 2)
            Der_r_vp_u = temp1 + temp2 - temp3
            vp_nabla.append(Der_r_vp_u)

            # 对于vs的偏导数
            temp1 = -8 * np.sin(theta) ** 2 * (den_d - den_u) * (vs_u + vs_d)
            temp1 = temp1 / ((den_u + den_d) * (vp_u + vp_d) ** 2)
            temp2 = 16 * np.sin(theta) ** 2 * vs_u
            temp2 = temp2 / (vp_u + vp_d) ** 2
            Der_r_vs_u = temp1 + temp2
            vs_nabla.append(Der_r_vs_u)

            # 对于rho的偏导数
            temp1 = -2 * den_d / (den_u + den_d) ** 2
            temp2 = 8 * np.sin(theta) ** 2 * den_d * (vs_u + vs_d) ** 2
            temp2 = temp2 / ((den_u + den_d) ** 2 * (vp_u + vp_d) ** 2)
            Der_r_den_u = temp1 + temp2
            rho_nabla.append(Der_r_den_u)

        if not self.multi:
            vp_nabla = np.vstack((np.array(vp_nabla), np.zeros(len(theta)))).T
            vs_nabla = np.vstack((np.array(vs_nabla), np.zeros(len(theta)))).T
            rho_nabla = np.vstack((np.array(rho_nabla), np.zeros(len(theta)))).T

            jac1 = np.vstack([np.diag(ivp) for ivp in vp_nabla])
            jac2 = np.vstack([np.diag(ivs) for ivs in vs_nabla])
            jac3 = np.vstack([np.diag(ir) for ir in rho_nabla])
            jac = np.hstack((jac1, jac2, jac3))
            G_mat = np.zeros(jac.shape)
            for n in range(jac.shape[0]):
                for m in range(3):
                    G_mat[n, (m) * 100: (m + 1) * 100] = (wav_mtx @ jac[n, m * 100: (m + 1) * 100].T).T
            W = block_diag(*[wav_mtx for _ in range(len(theta))])
            jac = W @ jac

        else:
            vp_nabla = np.vstack((np.array(vp_nabla), np.zeros((1, self.ntraces, len(theta)))))
            vs_nabla = np.vstack((np.array(vs_nabla), np.zeros((1, self.ntraces, len(theta)))))
            rho_nabla = np.vstack((np.array(rho_nabla), np.zeros((1, self.ntraces, len(theta)))))

            jac1 = np.stack([np.vstack([np.diag(vp_nabla[:, i, j]) for j in range(len(theta))])
                             for i in range(self.ntraces)])
            jac2 = np.stack([np.vstack([np.diag(vs_nabla[:, i, j]) for j in range(len(theta))])
                             for i in range(self.ntraces)])
            jac3 = np.stack([np.vstack([np.diag(rho_nabla[:, i, j]) for j in range(len(theta))])
                             for i in range(self.ntraces)])
            jac = np.concatenate((jac1, jac2, jac3), axis=2)
            # G_mat = np.zeros(jac.shape)
            # for n in range(jac.shape[0]):
            #     for m in range(3):
            #         G_mat[n, (m) * 100: (m + 1) * 100] = (wav_mtx @ jac[n, m * 100: (m + 1) * 100].T).T
            W = block_diag(*[wav_mtx for _ in range(len(theta))])
            jac = W @ jac

        return jac

    def showresult(self, dt0, trace, theta):
        t = np.arange(self.layers) * dt0
        plt.figure()
        if self.multi:
            plt.imshow(self.cal_data[trace], cmap="gray", extent=(theta[0], theta[-1], t[-1], t[0]))
            plt.title('synthetic data from Aki-Ri-Nonliner, trace = ' + str(trace))
        else:
            plt.imshow(self.cal_data, cmap="gray", extent=(theta[0], theta[-1], t[-1], t[0]))
            plt.title('synthetic data from Aki-Ri-Nonliner')
        plt.axis("tight")
        plt.xlabel('incident angle')
        plt.ylabel('time')
