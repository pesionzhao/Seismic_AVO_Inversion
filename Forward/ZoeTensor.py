"""
@File    : ZoeTensor.py
@Author  : Pesion
@Date    : 2023/11/21
@Desc    : 
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from Forward.BasicForward import ForwardModel
import torch
from torch import sin, cos
import numpy
import matplotlib.pyplot as plt

# import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZoeppritzTensor(torch.nn.Module):
    def __init__(self, ntraces, layers):
        super(ZoeppritzTensor).__init__()
        self.ntraces = ntraces
        self.layers = layers
        self.cal_layers = self.layers - 1
        # self.elements = torch.array(
        #     [
        #         ["PdPu", "SdPu", "PuPu", "SuPu"],
        #         ["PdSu", "SdSu", "PuSu", "SuSu"],
        #         ["PdPd", "SdPd", "PuPd", "SuPd"],
        #         ["PdSd", "SdSd", "PuSd", "SuSd"],
        #     ]
        # )
        self.elements = numpy.array(
            [
                ["PdPu"],
                ["PdSu"],
                ["PdPd"],
                ["PdSd"],
            ]
        )

    def __str__(self):
        return '正演方法为Zoeppritz'

    def forward_single_layer(self, a1, b1, rho1, a2, b2, rho2, theta):  # 单层的反射系数正演
        """
        通过Zoeppritz方程获取单一层面的层的反射系数

        Args:
            a1: 上层vp
            b1: 上层vs
            rho1: 上层密度
            a2: 下层vp
            b2: 下层vs
            rho2: 下层密度
            irfwav:
            ipol:
            theta: 入射角度集,角度制！！！

        Returns:
            当前层反射系数
            [
            ["PdPu", "SdPu", "PuPu", "SuPu"],
            ["PdSu", "SdSu", "PuSu", "SuSu"],
            ["PdPd", "SdPd", "PuPd", "SuPd"],
            ["PdSd", "SdSd", "PuSd", "SuSd"],
            ]

        """

        if self.ntraces != 1:
            a1 = a1[:, None]
            a2 = a2[:, None]
            b1 = b1[:, None]
            b2 = b2[:, None]
            rho1 = rho1[:, None]
            rho2 = rho2[:, None]

        # snell定理求解反射系数P，求解反射折射的角度
        p = torch.sin(theta) / a1
        if self.ntraces != 1:
            theta = torch.stack([*[theta for _ in range(self.ntraces)]])
        theta2 = torch.asin(torch.clamp(p * a2, min=-1, max=1))  # Trans. angle of P-wave
        phi1 = torch.asin(torch.clamp(p * b1, min=-1, max=1))  # Refl. angle of converted S-wave
        phi2 = torch.asin(torch.clamp(p * b2, min=-1, max=1))  # Trans. angle of converted S-wave

        # Matrix form of Zoeppritz equation
        M = torch.stack(  # shape:[4,4,theta]
            [
                torch.stack([-sin(theta), -cos(phi1), sin(theta2), cos(phi2)]),
                torch.stack([cos(theta), -sin(phi1), cos(theta2), -sin(phi2)]),
                torch.stack([
                    2 * rho1 * b1 * sin(phi1) * cos(theta),
                    rho1 * b1 * (1 - 2 * sin(phi1) ** 2),
                    2 * rho2 * b2 * sin(phi2) * cos(theta2),
                    rho2 * b2 * (1 - 2 * sin(phi2) ** 2)
                ]),
                torch.stack([
                    (-rho1) * a1 * (1 - 2 * sin(phi1) ** 2),
                    rho1 * b1 * sin(2 * phi1),
                    rho2 * a2 * (1 - 2 * sin(phi2) ** 2),
                    (-rho2) * b2 * sin(2 * phi2)
                ]),
            ]
        )
        N = torch.stack(  # shape:[4,1,theta]
            [
                torch.stack([sin(theta), cos(phi1), -sin(theta2), -cos(phi2)]),
                torch.stack([cos(theta), -sin(phi1), cos(theta2), -sin(phi2)]),
                torch.stack([
                    2 * rho1 * b1 * sin(phi1) * cos(theta),
                    rho1 * b1 * (1 - 2 * sin(phi1) ** 2),
                    2 * rho2 * b2 * sin(phi2) * cos(theta2),
                    rho2 * b2 * (1 - 2 * sin(phi2) ** 2),
                ]),
                torch.stack([
                    rho1 * a1 * (1 - 2 * sin(phi1) ** 2),
                    -rho1 * b1 * sin(2 * phi1),
                    -rho2 * a2 * (1 - 2 * sin(phi2) ** 2),
                    rho2 * b2 * sin(2 * phi2),
                ]),
            ],
        )

        # N = torch.array(  # shape:[4,1,theta]
        #     [
        #         [sin(theta)],
        #         [cos(theta)],
        #         [
        #             2 * rho1 * b1 * sin(phi1) * cos(theta)
        #         ],
        #         [
        #             rho1 * a1 * (1 - 2 * sin(phi1) ** 2)
        #         ],
        #     ],
        #     dtype="float",
        # )

        # Create Zoeppritz coefficient for all angles
        coef = torch.zeros((4, 4, M.shape[-1]), device=device) if self.ntraces == 1 \
            else torch.zeros((M.shape[-2], 4, 4, M.shape[-1]), device=device)  # [4,4,(layer),theta]
        for i in range(M.shape[-1]):  # 按角度数遍历
            Mi = M[..., i]  # 第i个角度的所有
            Ni = N[..., i]
            if self.ntraces == 1:
                icoef = torch.dot(torch.linalg.inv(Mi), Ni)  # M^-1 点乘 N
            else:
                # icoef = torch.stack([torch.dot(torch.linalg.inv(Mi[..., j]), Ni[..., j])
                # for j in range(self.ntraces)], axis=-1) # for循环求每个子矩阵的逆，下一条语句为直接求三维的逆
                icoef = torch.linalg.inv(torch.permute(Mi, [2, 0, 1])) @ torch.permute(Ni, [2, 0, 1])  # [ntraces, 4, 4]
            coef[..., i] = icoef

        return coef

    def forward_mutil(self, vp, vs, rho, theta, element="PdPu", show=False):  # 多层反射系数正演
        """
        获取所有层的反射系数

        Args:
            vp: 纵波序列
            vs: 横波序列
            rho: 密度序列
            theta: 角度集
            element: str
            ["PdPu", "SdPu", "PuPu", "SuPu"],
            ["PdSu", "SdSu", "PuSu", "SuSu"],
            ["PdPd", "SdPd", "PuPd", "SuPd"],
            ["PdSd", "SdSd", "PuSd", "SuSd"],

        Returns:
            通过for循环得到所有层反射系数

        """
        if len(vp) != len(vs) or len(vp) != len(rho):
            raise ValueError("vp, vs ,rho的长度不一样，请检查代码")
        layer = len(vp) - 1  # 由于计算单个层需要用到上下两个层的属性，所以最终计算出的反射系数层数比vp的采样数少一
        ntheta = len(theta)
        elements = numpy.array(
            [
                ["PdPu", "SdPu", "PuPu", "SuPu"],
                ["PdSu", "SdSu", "PuSu", "SuSu"],
                ["PdPd", "SdPd", "PuPd", "SuPd"],
                ["PdSd", "SdSd", "PuSd", "SuSd"],
            ]
        )
        coef = torch.zeros((layer + 1, ntheta), device=device)
        for i in range(layer):
            a1 = vp[i]
            a2 = vp[i + 1]
            b1 = vs[i]
            b2 = vs[i + 1]
            rho1 = rho[i]
            rho2 = rho[i + 1]

            icoef = self.forward_single_layer(a1, b1, rho1, a2, b2, rho2, theta)
            index = numpy.where(elements == element)
            coef[i, ...] = torch.squeeze(icoef[index])
        return coef

    def forward_mutiltraces(self, vp, vs, rho, theta, element="PdPu"):  # 多层反射系数正演
        """
        获取所有层的反射系数

        Args:
            vp: 纵波序列
            vs: 横波序列
            rho: 密度序列
            theta: 角度集
            element: str
            ["PdPu", "SdPu", "PuPu", "SuPu"],
            ["PdSu", "SdSu", "PuSu", "SuSu"],
            ["PdPd", "SdPd", "PuPd", "SuPd"],
            ["PdSd", "SdSd", "PuSd", "SuSd"],

        Returns:
            Rpp: 所有层反射系数，大小为[traces, layers, Ntheta]

        Notes:
            此方法计算单个层的反射系数通过for循环拼接在一起，不使用for循环的矩阵计算由下一个函数实现，但速度没有得到提升

        """
        layer = len(vp) - 1  # 由于计算单个层需要用到上下两个层的属性，所以最终计算出的反射系数层数比vp的采样数少一
        ntheta = len(theta)
        coef = torch.zeros((self.ntraces, layer + 1, ntheta), device=device)
        for i in range(layer):
            a1 = vp[i]
            a2 = vp[i + 1]
            b1 = vs[i]
            b2 = vs[i + 1]
            rho1 = rho[i]
            rho2 = rho[i + 1]

            icoef = self.forward_single_layer(a1, b1, rho1, a2, b2, rho2, theta)
            index = numpy.where(self.elements == element)
            coef[:, i, :] = torch.squeeze(icoef[:, index[0], index[1], :])
        return coef

    # def forward_mutiltracesAndlayers(self, vp, vs, rho, theta):
    #     """
    #
    #     Args:
    #         vp: [layer, trace]
    #         vs: [layer, trace]
    #         rho: [layer, trace]
    #         theta: 入射角度集
    #
    #     Returns:
    #         Rpp: 所有层所有道的反射系数
    #
    #     """
    #     a1 = vp[:-1]
    #     a2 = vp[1:]
    #     b1 = vs[:-1]
    #     b2 = vs[1:]
    #     rho1 = rho[:-1]
    #     rho2 = rho[1:]
    #     if self.ntraces != 1:
    #         a1 = a1[..., torch.newaxis]
    #         a2 = a2[..., torch.newaxis]
    #         b1 = b1[..., torch.newaxis]
    #         b2 = b2[..., torch.newaxis]
    #         rho1 = rho1[..., torch.newaxis]
    #         rho2 = rho2[..., torch.newaxis]
    #
    #     p = torch.sin(theta) / a1
    #     if self.ntraces != 1:
    #         theta = torch.stack([*[theta for _ in range(self.ntraces)]])
    #     theta = torch.stack([*[theta for _ in range(self.cal_layers)]])
    #     theta2 = torch.vectorize(cm.asin)(p * a2) # Trans. angle of P-wave
    #     phi1 = torch.vectorize(cm.asin)(p * b1) # Refl. angle of converted S-wave
    #     phi2 = torch.vectorize(cm.asin)(p * b2) # Trans. angle of converted S-wave
    #
    #     # Matrix form of Zoeppritz equation
    #
    #     M = torch.array(  # shape:[4,4,theta]
    #         [
    #             [-sin(theta), -cos(phi1), sin(theta2), cos(phi2)],
    #             [cos(theta), -sin(phi1), cos(theta2), -sin(phi2)],
    #             [
    #                 2 * rho1 * b1 * sin(phi1) * cos(theta),
    #                 rho1 * b1 * (1 - 2 * sin(phi1) ** 2),
    #                 2 * rho2 * b2 * sin(phi2) * cos(theta2),
    #                 rho2 * b2 * (1 - 2 * sin(phi2) ** 2),
    #             ],
    #             [
    #                 (-rho1) * a1 * (1 - 2 * sin(phi1) ** 2),
    #                 rho1 * b1 * sin(2 * phi1),
    #                 rho2 * a2 * (1 - 2 * sin(phi2) ** 2),
    #                 (-rho2) * b2 * sin(2 * phi2),
    #             ],
    #         ],
    #         dtype="complex64",
    #     )
    #     N = torch.array(  # shape:[4,1,theta]
    #         [
    #             [sin(theta)],
    #             [cos(theta)],
    #             [2 * rho1 * b1 * sin(phi1) * cos(theta)],
    #             [rho1 * a1 * (1 - 2 * sin(phi1) ** 2)],
    #         ],
    #         dtype="complex64",
    #     )
    #
    #     # Create Zoeppritz coefficient for all angles
    #     coef = (torch.linalg.inv(torch.transpose(M, [2, 3, 4, 0, 1])) @ torch.transpose(N, [2, 3, 4, 0, 1]))[..., 0, 0]
    #     coef = torch.vstack([coef, torch.zeros((1, coef.shape[1], coef.shape[2]))])
    #     coef = torch.transpose(coef, [1, 0, 2])
    #
    #     return coef.real

    def forward(self, vp, vs, rho, theta, wavemat):
        wavemat = torch.tensor(wavemat, dtype=torch.float32, device=device)
        vp = torch.tensor(vp, device=device)
        vs = torch.tensor(vs, device=device)
        rho = torch.tensor(rho, device=device)
        theta = torch.tensor(theta, device=device)
        if self.ntraces == 1:
            ref = self.forward_mutil(vp, vs, rho, theta)
            cal_data = wavemat @ ref
        # else: # 通过比较for循环与矩阵计算的时间
        #     self.cal_data = torch.zeros((self.ntraces, self.layers, len(theta)))
        #     ntraces = self.ntraces
        #     self.ntraces=1
        #     for i in range(ntraces):
        #         vp_layer = vp[:, i]
        #         vs_layer = vs[:, i]
        #         rho_layer = rho[:, i]
        #         ref = self.forward_mutil(vp_layer, vs_layer, rho_layer, theta)
        #         self.cal_data[i, :, :] = wavemat @ ref
        else:
            ref = self.forward_mutiltraces(vp, vs, rho, theta)
            # ref = self.forward_mutiltracesAndlayers(vp, vs, rho, theta) # 通过矩阵同时计算所有道所有层，未能达到提速
            wavemat = torch.stack([*[wavemat for _ in range(self.ntraces)]])
            cal_data = wavemat @ ref

        return cal_data

    def showresult(self, dt0, trace, theta):
        t = torch.arange(self.layers) * dt0
        plt.figure()
        if self.ntraces == 1:
            plt.imshow(self.cal_data, cmap="gray", extent=(theta[0], theta[-1], t[-1], t[0]))
            plt.title('seismic data from Zoeppritz')
        else:
            plt.imshow(self.cal_data[trace], cmap="gray", extent=(theta[0], theta[-1], t[-1], t[0]))
            plt.title('seismic data from Zoeppritz, trace = %d' % trace)
        plt.axis("tight")
        plt.xlabel('incident angle')
        plt.ylabel('time')

    def jacobian(self, vp, vs, rho, theta, wav_mtx):
        jac = self.GenerateGrad_multitrace(vp, vs, rho, theta, wav_mtx)
        if self.ntraces == 1:
            jac = jac.squeeze()
        return jac

    def GenerateGrad(self, vp, vs, rho, theta):
        Nt = self.layers
        Ntheta = torch.size(theta, 0)
        Theta_use = theta[-1]

        ## Version 1 -> Use all angle
        # G_mat = torch.zeros((Nt*Ntheta, 3 * Nt))
        #
        # for n in range(Nt-1):
        #     for m in range(1, Ntheta+1):
        #
        #         G_row     = (m-1) * Nt + n
        #         G_col_vp  = n
        #         G_col_vs  = n + Nt
        #         G_col_den = n + 2 * Nt
        #         Grad_cal_vp1, Grad_cal_vs1, Grad_cal_rho1 = Grad_cal(vp[n], vp[n+1], vs[n], vs[n+1], den[n], den[n+1], Theta[m-1])
        #
        #         G_mat[G_row, G_col_vp] = Grad_cal_vp1
        #         G_mat[G_row, G_col_vs] = Grad_cal_vs1
        #         G_mat[G_row, G_col_den] = Grad_cal_rho1

        ## Version 2 -> Just use single angle
        G_mat = torch.zeros((Nt, 3 * Nt))

        for n in range(Nt - 1):
            for m in range(1, Ntheta + 1):
                G_row = n
                G_col_vp = n
                G_col_vs = n + Nt
                G_col_den = n + 2 * Nt
                Grad_cal_vp1, Grad_cal_vs1, Grad_cal_rho1 = self.Grad_cal_multitrace(vp[n], vp[n + 1], vs[n],
                                                                                     vs[n + 1],
                                                                                     rho[n],
                                                                                     rho[n + 1], Theta_use)

                G_mat[G_row, G_col_vp] = Grad_cal_vp1
                G_mat[G_row, G_col_vs] = Grad_cal_vs1
                G_mat[G_row, G_col_den] = Grad_cal_rho1

        return G_mat

    def GenerateGrad_multitrace(self, vp, vs, rho, theta, wav_mtx):
        """

        Args:
            vp: [layers, traces]
            vs: [layers, traces]
            rho: [layers, traces]
            theta: 入射角集
            wav_mtx: 子波矩阵，用于与Rpp的导数点乘得到地震数据关于m的偏导数

        Returns:
            地震数据关于[vp,vs,rho]的偏导数，大小为[traces, Ntheta*layers, 3*layers]

        Notes:
            解释雅可比矩阵大小为[traces, Ntheta*layers, 3*layers]：
            在之后的迭代计算时将vp,vs,rho拼接在了一起，地震数据也将不同角度的数据拼接在一起，
            故叠前参数为[traces, 3*layers]，地震数据为[traces, Ntheta*layers]
            根据雅克比矩阵定义，大小为[traces, Ntheta*layers, 3*layers]

        """
        Ntheta = len(theta)
        G_mat = torch.zeros((self.ntraces, Ntheta * self.layers, 3 * self.layers))

        # # 注释的代码表示用for循环堆叠构建G_mat, 用jac表示，由于计算速度不如赋值法，故没有使用
        # vp1_nabla = []
        # vs1_nabla = []
        # rho1_nabla = []

        for n in range(self.cal_layers):
            # 只取上界面速度的偏导数
            Grad_cal_vp1, Grad_cal_vs1, Grad_cal_rho1 = self.Grad_cal_multitrace(vp[n], vp[n + 1], vs[n], vs[n + 1],
                                                                                 rho[n],
                                                                                 rho[n + 1], theta)

            # vp1_nabla.append(Grad_cal_vp1)
            # vs1_nabla.append(Grad_cal_vs1)
            # rho1_nabla.append(Grad_cal_rho1)
            for i in range(Ntheta):
                G_mat[:, n + i * self.layers, n] = Grad_cal_vp1[..., i]
                G_mat[:, n + i * self.layers, n + self.layers] = Grad_cal_vs1[..., i]
                G_mat[:, n + i * self.layers, n + 2 * self.layers] = Grad_cal_rho1[..., i]

        # vp_nabla = torch.vstack((torch.array(vp1_nabla), torch.zeros((1, self.ntraces, len(theta)))))
        # vs_nabla = torch.vstack((torch.array(vs1_nabla), torch.zeros((1, self.ntraces, len(theta)))))
        # rho_nabla = torch.vstack((torch.array(rho1_nabla), torch.zeros((1, self.ntraces, len(theta)))))
        #
        # jac1 = torch.stack([torch.vstack([torch.diag(vp_nabla[:, i, j]) for j in range(len(theta))])
        #                  for i in range(self.ntraces)])
        # jac2 = torch.stack([torch.vstack([torch.diag(vs_nabla[:, i, j]) for j in range(len(theta))])
        #                  for i in range(self.ntraces)])
        # jac3 = torch.stack([torch.vstack([torch.diag(rho_nabla[:, i, j]) for j in range(len(theta))])
        #                  for i in range(self.ntraces)])
        # jac = torch.concatenate((jac1, jac2, jac3), axis=2)
        # W = block_diag(*[wav_mtx for _ in range(Ntheta)])
        # G_mat1 = W @ G_mat
        W = torch.block_diag(*[wav_mtx.T for _ in range(3)])
        # import pdb
        # pdb.set_trace()
        G_mat = G_mat @ W

        # G_mat2 = torch.zeros(G_mat.shape)
        # for n in range(G_mat.shape[0]):
        #     for i in range(G_mat.shape[1]):
        #         for m in range(3):
        #             G_mat2[n, i, (m) * 100: (m + 1) * 100] = (wav_mtx @ G_mat[n, i, m * 100: (m + 1) * 100])

        return G_mat

    def Grad_cal_multitraceAndlayer(self, vp, vs, rho, theta):
        # TODO 尝试一下同时计算所有层
        pass

    def Grad_cal(self, vp1, vp2, vs1, vs2, rho1, rho2, Alpha):
        """
        Autor: 粱兴城
        """

        # 该函数用来对所有输入参数求偏导数
        # if vp1 == 0. or vp2 == 0. or vs1 == 0. or vs2 == 0. or rho1 == 0. or rho2 == 0.:
        #
        #     return 0., 0., 0.
        #
        # else:

        # Snell定理
        p = vp1 / torch.sin(Alpha)
        Beta = torch.arcsin(vs1 / p)
        Alpha_1 = torch.arcsin(vp2 / p)
        Beta_1 = torch.arcsin(vs2 / p)

        # Zoeppritz方程中A的定义
        A = torch.array([
            [torch.sin(Alpha), torch.cos(Beta), -torch.sin(Alpha_1), torch.cos(Beta_1)],

            [torch.cos(Alpha), -torch.sin(Beta), torch.cos(Alpha_1), torch.sin(Beta_1)],

            [torch.cos(2 * Beta), -(vs1 / vp1) * torch.sin(2 * Beta),
             -(rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1),
             -(rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)],

            [(vs1 ** 2 / vp1) * torch.sin(2 * Alpha), vs1 * torch.cos(2 * Beta),
             (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1), -(rho2 / rho1) * vs2 * torch.cos(2 * Beta_1)]
        ], dtype='float')

        # Zoeppritz方程中b的定义
        C = torch.array(
            [torch.sin(Alpha), torch.cos(Alpha), -torch.cos(2 * Beta), (vs1 ** 2 / vp1) * torch.sin(2 * Alpha)],
            dtype='float')
        # C = C.reshape(C.shape[0], 1)

        # Zoeppritz方程中A的偏导数定义  AC=b
        A_Vp1 = (1 / vp1) * torch.array([
            [0, torch.tan(Beta) * torch.sin(Beta), torch.sin(Alpha_1), torch.tan(Beta_1) * torch.sin(Beta_1)],

            [0, torch.sin(Beta), torch.tan(Alpha_1) * torch.sin(Alpha_1), -torch.sin(Beta_1)],

            [2 * (1 - torch.cos(2 * Beta)), -((torch.tan(Beta)) ** 2 - 2) * (vs1 / vp1) * torch.sin(2 * Beta),
             (rho2 / rho1) * (vp2 / vp1) * (3 * torch.cos(2 * Beta_1) - 2),
             (2 - (torch.tan(Beta_1)) ** 2) * (rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)],

            [-(vs1 ** 2 / vp1) * torch.sin(2 * Alpha), 2 * vs1 * (1 - torch.cos(2 * Beta)),
             ((torch.tan(Alpha_1)) ** 2 - 1) * (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
             (rho2 / rho1) * vs2 * 2 * (torch.cos(2 * Beta_1) - 1)]
        ], dtype='float')

        A_Vp2 = (1 / vp2) * torch.array([
            [0, 0, -torch.sin(Alpha_1), 0],

            [0, 0, -torch.tan(Alpha_1) * torch.sin(Alpha_1), 0],

            [0, 0, -(rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1), 0],

            [0, 0, (rho2 / rho1) * (vs2 ** 2 / vp2) * (-2 * torch.tan(Alpha_1) * (torch.sin(Alpha_1) ** 2)), 0]
        ], dtype='float')

        A_Vs1 = (1 / vs1) * torch.array([
            [0, -torch.tan(Beta) * torch.sin(Beta), 0, 0],

            [0, -torch.sin(Beta), 0, 0],

            [- 4 * (torch.sin(Beta)) ** 2, ((torch.tan(Beta)) ** 2 - 2) * (vs1 / vp1) * torch.sin(2 * Beta), 0, 0],

            [2 * (vs1 ** 2 / vp1) * torch.sin(2 * Alpha), vs1 * (1 - 6 * (torch.sin(Beta) ** 2)), 0, 0]
        ], dtype='float')

        A_Vs2 = (1 / vs2) * torch.array([
            [0, 0, 0, -torch.tan(Beta_1) * torch.sin(Beta_1)],

            [0, 0, 0, torch.sin(Beta_1)],

            [0, 0, 2 * (rho2 / rho1) * (vp2 / vp1) * (1 - torch.cos(2 * Beta_1)),
             (rho2 / rho1) * (vs2 / vp1) * ((torch.tan(Beta_1) ** 2) - 2) * torch.sin(2 * Beta_1)],

            [0, 0, 2 * (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
             vs2 * (rho2 / rho1) * (6 * (torch.sin(Beta_1) ** 2) - 1)]
        ], dtype='float')

        A_Rho1 = (1 / rho1) * torch.array([
            [0, 0, 0, 0],

            [0, 0, 0, 0],

            [0, 0, (rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1),
             (rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)],

            [0, 0, -(rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
             (rho2 / rho1) * vs2 * torch.cos(2 * Beta_1)]
        ], dtype='float')

        A_Rho2 = (1 / rho2) * torch.array([
            [0, 0, 0, 0],

            [0, 0, 0, 0],

            [0, 0, -(rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1),
             -(rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)],

            [0, 0, (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
             -(rho2 / rho1) * vs2 * torch.cos(2 * Beta_1)]
        ], dtype='float')

        # Zoeppritz方程中C的偏导数定义
        C_Vp1 = (1 / vp1) * torch.array([0, 0, -4 * (torch.sin(Beta) ** 2), -(vs1 ** 2 / vp1) * torch.sin(2 * Alpha)],
                                        dtype='float')
        # C_Vp1 = C_Vp1.reshape(C_Vp1.shape[0], 1)

        C_Vp2 = (1 / vp2) * torch.array([0, 0, 0, 0], dtype='float')
        # C_Vp2 = C_Vp2.reshape(C_Vp2.shape[0], 1)

        C_Vs1 = (1 / vs1) * torch.array([0, 0, 4 * (torch.sin(Beta) ** 2), 2 * (vs1 ** 2 / vp1) * torch.sin(2 * Alpha)],
                                        dtype='float')
        # C_Vs1 = C_Vs1.reshape(C_Vs1.shape[0], 1)

        C_Vs2 = (1 / vs2) * torch.array([0, 0, 0, 0], dtype='float')
        # C_Vs2 = C_Vs2.reshape(C_Vs2.shape[0], 1)

        C_Rho1 = torch.array([0, 0, 0, 0], dtype='float')
        # C_Rho1 = C_Rho1.reshape(C_Rho1.shape[0], 1)

        C_Rho2 = torch.array([0, 0, 0, 0], dtype='float')
        C_Rho2 = C_Rho2.reshape(C_Rho2.shape[0], 1)

        # 1. A-1*C按照反射透射系数全定义
        A_pinv = torch.linalg.pinv(A)
        Rpp_grad = torch.dot(A_pinv, C)

        # # 2. A-1*C仅按照Rpp定义
        # A_pinv = torch.linalg.pinv(A)
        # Rpp_grad = torch.array([Rpp, 0, 0, 0])
        # Rpp_grad = Rpp_grad.reshape(Rpp_grad.shape[0], 1)

        Grad_cal_vp1 = C_Vp1 - torch.dot(A_Vp1, Rpp_grad)
        # import pdb
        # pdb.set_trace()
        Grad_cal_vp1 = torch.dot(A_pinv, Grad_cal_vp1)[0].real

        # Grad_cal_vp2 = C_Vp2 - torch.dot(A_Vp2, Rpp_grad)
        # Grad_cal_vp2 = torch.dot(A_pinv, Grad_cal_vp2)[0].real

        Grad_cal_vs1 = C_Vs1 - torch.dot(A_Vs1, Rpp_grad)
        Grad_cal_vs1 = torch.dot(A_pinv, Grad_cal_vs1)[0].real

        # Grad_cal_vs2 = C_Vs2 - torch.dot(A_Vs2, Rpp_grad)
        # Grad_cal_vs2 = torch.dot(A_pinv, Grad_cal_vs2)[0].real

        Grad_cal_rho1 = C_Rho1 - torch.dot(A_Rho1, Rpp_grad)
        Grad_cal_rho1 = torch.dot(A_pinv, Grad_cal_rho1)[0].real

        # Grad_cal_rho2 = C_Rho2 - torch.dot(A_Rho2, Rpp_grad)
        # Grad_cal_rho2 = torch.dot(A_pinv, Grad_cal_rho2)[0].real

        return Grad_cal_vp1, Grad_cal_vs1, Grad_cal_rho1

    def Grad_cal_multitrace(self, vp1, vp2, vs1, vs2, rho1, rho2, Alpha):
        """
        Args:
            vp1: 上层vp，大小为[traces,]
            vp2: 下层vp, 大小为[traces,]
            vs1: 上层vs, 大小为[traces,]
            vs2: 下层vs, 大小为[traces,]
            rho1: 上层rho, 大小为[traces,]
            rho2: 下层rho, 大小为[traces,]
            Alpha: 输入角度, 不能为列表，必须为矩阵

        Returns:
            {
                Grad_cal_vp1: 单个层的Rpp对vp1的偏导数，大小为[traces, Ntheta]
                Grad_cal_vs1: 单个层的Rpp对vs1的偏导数，大小为[traces, Ntheta]
                Grad_cal_rho1: 单个层的Rpp对rho1的偏导数，大小为[traces, Ntheta]
            }

        Notes:
            由于单一层的Rpp只与穿过该层的上层速度与下层速度有关，所以该层的反射系数对其他层的参数的偏导数为0，
            这里未进行说明，需要在调用此函数时处理，详见函数GenerateGrad_multitrace(),也就是说Grad_cal_vp1的大小应为[traces, Ntheta*layers]

        """

        # 该函数用来对所有输入参数求偏导数
        # if vp1 == 0. or vp2 == 0. or vs1 == 0. or vs2 == 0. or rho1 == 0. or rho2 == 0.:
        #
        #     return 0., 0., 0.
        #
        # else:

        if self.ntraces != 1:
            vp1 = vp1[:, torch.newaxis]
            vp2 = vp2[:, torch.newaxis]
            vs1 = vs1[:, torch.newaxis]
            vs2 = vs2[:, torch.newaxis]
            rho1 = rho1[:, torch.newaxis]
            rho2 = rho2[:, torch.newaxis]

        # Snell定理
        p = torch.sin(Alpha) / vp1
        if self.ntraces != 1:
            Alpha = torch.stack([*[Alpha for _ in range(self.ntraces)]])
        Alpha_1 = torch.asin(torch.clamp(p * vp2, min=-1, max=1))  # Trans. angle of P-wave
        Beta = torch.asin(torch.clamp(p * vs1, min=-1, max=1))  # Refl. angle of converted S-wave
        Beta_1 = torch.asin(torch.clamp(p * vs2, min=-1, max=1))  # Trans. angle of converted S-wave

        # Zoeppritz方程中A的定义
        A = torch.stack([
            torch.stack([torch.sin(Alpha), torch.cos(Beta), -torch.sin(Alpha_1), torch.cos(Beta_1)]),

            torch.stack([torch.cos(Alpha), -torch.sin(Beta), torch.cos(Alpha_1), torch.sin(Beta_1)]),

            torch.stack([torch.cos(2 * Beta), -(vs1 / vp1) * torch.sin(2 * Beta),
                         -(rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1),
                         -(rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)]),

            torch.stack([(vs1 ** 2 / vp1) * torch.sin(2 * Alpha), vs1 * torch.cos(2 * Beta),
                         (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
                         -(rho2 / rho1) * vs2 * torch.cos(2 * Beta_1)])
        ])

        # Zoeppritz方程中b的定义
        C = torch.stack(
            [torch.sin(Alpha), torch.cos(Alpha), -torch.cos(2 * Beta), (vs1 ** 2 / vp1) * torch.sin(2 * Alpha)])
        # C = C.reshape(C.shape[0], 1)

        zero_element = torch.zeros(Alpha.shape, device=device)

        # Zoeppritz方程中A的偏导数定义  AC=b
        A_Vp1 = (1 / vp1) * torch.stack([
            torch.stack([zero_element, torch.tan(Beta) * torch.sin(Beta), torch.sin(Alpha_1),
                         torch.tan(Beta_1) * torch.sin(Beta_1)]),

            torch.stack([zero_element, torch.sin(Beta), torch.tan(Alpha_1) * torch.sin(Alpha_1), -torch.sin(Beta_1)]),

            torch.stack(
                [2 * (1 - torch.cos(2 * Beta)), -((torch.tan(Beta)) ** 2 - 2) * (vs1 / vp1) * torch.sin(2 * Beta),
                 (rho2 / rho1) * (vp2 / vp1) * (3 * torch.cos(2 * Beta_1) - 2),
                 (2 - (torch.tan(Beta_1)) ** 2) * (rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)]),

            torch.stack([-(vs1 ** 2 / vp1) * torch.sin(2 * Alpha), 2 * vs1 * (1 - torch.cos(2 * Beta)),
                         ((torch.tan(Alpha_1)) ** 2 - 1) * (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
                         (rho2 / rho1) * vs2 * 2 * (torch.cos(2 * Beta_1) - 1)])
        ])

        # A_Vp2 = (1 / vp2) * torch.array([
        #     [0, 0, -torch.sin(Alpha_1), 0],
        #
        #     [0, 0, -torch.tan(Alpha_1) * torch.sin(Alpha_1), 0],
        #
        #     [0, 0, -(rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1), 0],
        #
        #     [0, 0, (rho2 / rho1) * (vs2 ** 2 / vp2) * (-2 * torch.tan(Alpha_1) * (torch.sin(Alpha_1) ** 2)), 0]
        # ], dtype='float')

        A_Vs1 = (1 / vs1) * torch.stack([
            torch.stack([zero_element, -torch.tan(Beta) * torch.sin(Beta), zero_element, zero_element]),

            torch.stack([zero_element, -torch.sin(Beta), zero_element, zero_element]),

            torch.stack([- 4 * (torch.sin(Beta)) ** 2, ((torch.tan(Beta)) ** 2 - 2) * (vs1 / vp1) * torch.sin(2 * Beta),
                         zero_element,
                         zero_element]),

            torch.stack(
                [2 * (vs1 ** 2 / vp1) * torch.sin(2 * Alpha), vs1 * (1 - 6 * (torch.sin(Beta) ** 2)), zero_element,
                 zero_element])
        ])

        # A_Vs2 = (1 / vs2) * torch.array([
        #     [0, 0, 0, -torch.tan(Beta_1) * torch.sin(Beta_1)],
        #
        #     [0, 0, 0, torch.sin(Beta_1)],
        #
        #     [0, 0, 2 * (rho2 / rho1) * (vp2 / vp1) * (1 - torch.cos(2 * Beta_1)),
        #      (rho2 / rho1) * (vs2 / vp1) * ((torch.tan(Beta_1) ** 2) - 2) * torch.sin(2 * Beta_1)],
        #
        #     [0, 0, 2 * (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
        #      vs2 * (rho2 / rho1) * (6 * (torch.sin(Beta_1) ** 2) - 1)]
        # ], dtype='float')

        A_Rho1 = (1 / rho1) * torch.stack([
            torch.stack([zero_element, zero_element, zero_element, zero_element]),

            torch.stack([zero_element, zero_element, zero_element, zero_element]),

            torch.stack([zero_element, zero_element, (rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1),
                         (rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)]),

            torch.stack([zero_element, zero_element, -(rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
                         (rho2 / rho1) * vs2 * torch.cos(2 * Beta_1)])
        ])

        # A_Rho2 = (1 / rho2) * torch.array([
        #     [0, 0, 0, 0],
        #
        #     [0, 0, 0, 0],
        #
        #     [0, 0, -(rho2 / rho1) * (vp2 / vp1) * torch.cos(2 * Beta_1),
        #      -(rho2 / rho1) * (vs2 / vp1) * torch.sin(2 * Beta_1)],
        #
        #     [0, 0, (rho2 / rho1) * (vs2 ** 2 / vp2) * torch.sin(2 * Alpha_1),
        #      -(rho2 / rho1) * vs2 * torch.cos(2 * Beta_1)]
        # ], dtype='float')

        # Zoeppritz方程中C的偏导数定义
        C_Vp1 = (1 / vp1) * torch.stack(
            [zero_element, zero_element, -4 * (torch.sin(Beta) ** 2), -(vs1 ** 2 / vp1) * torch.sin(2 * Alpha)])
        # C_Vp1 = C_Vp1.reshape(C_Vp1.shape[0], 1)

        # C_Vp2 = (1 / vp2) * torch.array([0, 0, 0, 0], dtype='float')
        # C_Vp2 = C_Vp2.reshape(C_Vp2.shape[0], 1)

        C_Vs1 = (1 / vs1) * torch.stack(
            [zero_element, zero_element, 4 * (torch.sin(Beta) ** 2), 2 * (vs1 ** 2 / vp1) * torch.sin(2 * Alpha)])
        # C_Vs1 = C_Vs1.reshape(C_Vs1.shape[0], 1)

        # C_Vs2 = (1 / vs2) * torch.array([0, 0, 0, 0], dtype='float')
        # C_Vs2 = C_Vs2.reshape(C_Vs2.shape[0], 1)

        C_Rho1 = torch.stack([zero_element, zero_element, zero_element, zero_element])
        # C_Rho1 = C_Rho1.reshape(C_Rho1.shape[0], 1)

        # C_Rho2 = torch.array([zero_element, zero_element, zero_element, zero_element], dtype='float')
        # C_Rho2 = C_Rho2.reshape(C_Rho2.shape[0], 1)

        if self.ntraces != 1:
            A = torch.permute(A, [2, 3, 0, 1])
            A_Vp1 = torch.permute(A_Vp1, [2, 3, 0, 1])
            A_Vs1 = torch.permute(A_Vs1, [2, 3, 0, 1])
            A_Rho1 = torch.permute(A_Rho1, [2, 3, 0, 1])

            C = torch.permute(C, [1, 2, 0])[..., None]
            C_Vp1 = torch.permute(C_Vp1, [1, 2, 0])[..., None]
            C_Vs1 = torch.permute(C_Vs1, [1, 2, 0])[..., None]
            C_Rho1 = torch.permute(C_Rho1, [1, 2, 0])[..., None]
        else:
            A = torch.permute(A, [2, 0, 1])
            A_Vp1 = torch.permute(A_Vp1, [2, 0, 1])
            A_Vs1 = torch.permute(A_Vs1, [2, 0, 1])
            A_Rho1 = torch.permute(A_Rho1, [2, 0, 1])

            C = torch.permute(C, [1, 0])[..., None]
            C_Vp1 = torch.permute(C_Vp1, [1, 0])[..., None]
            C_Vs1 = torch.permute(C_Vs1, [1, 0])[..., None]
            C_Rho1 = torch.permute(C_Rho1, [1, 0])[..., None]

        # 1. A-1*C按照反射透射系数全定义
        A_pinv = torch.linalg.pinv(A)
        Rpp_grad = A_pinv @ C

        # # 2. A-1*C仅按照Rpp定义
        # A_pinv = torch.linalg.pinv(A)
        # Rpp_grad = torch.array([Rpp, 0, 0, 0])
        # Rpp_grad = Rpp_grad.reshape(Rpp_grad.shape[0], 1)

        Grad_cal_vp1 = C_Vp1 - A_Vp1 @ Rpp_grad
        # import pdb
        # pdb.set_trace()
        Grad_cal_vp1 = (A_pinv @ Grad_cal_vp1)[..., 0, 0]

        # Grad_cal_vp2 = C_Vp2 - torch.dot(A_Vp2, Rpp_grad)
        # Grad_cal_vp2 = torch.dot(A_pinv, Grad_cal_vp2)[0].real

        Grad_cal_vs1 = C_Vs1 - A_Vs1 @ Rpp_grad
        Grad_cal_vs1 = (A_pinv @ Grad_cal_vs1)[..., 0, 0]

        # Grad_cal_vs2 = C_Vs2 - torch.dot(A_Vs2, Rpp_grad)
        # Grad_cal_vs2 = torch.dot(A_pinv, Grad_cal_vs2)[0].real

        Grad_cal_rho1 = C_Rho1 - A_Rho1 @ Rpp_grad
        Grad_cal_rho1 = (A_pinv @ Grad_cal_rho1)[..., 0, 0]

        # Grad_cal_rho2 = C_Rho2 - torch.dot(A_Rho2, Rpp_grad)
        # Grad_cal_rho2 = torch.dot(A_pinv, Grad_cal_rho2)[0].real

        return Grad_cal_vp1, Grad_cal_vs1, Grad_cal_rho1
