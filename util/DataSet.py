"""
@File :Gan_DataSet.py
@Author :Pesion
@Date :2023/9/12
@Desc : 
"""
import numpy as np
import scipy.io as scio
from abc import ABCMeta, abstractmethod
import math
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


class Train_DataSet():
    def __init__(self, vp, vs, rho, theta, use_trace, use_layer):
        """

        Args:
            vp: 二维时 shape应为[layers, trace]
            vs: 二维时 shape应为[layers, trace]
            rho: 二维时 shape应为[layers, trace]
            theta: 角度制的角度集
        """
        self.theta_rad = np.radians(theta)
        self.rho = rho[use_layer, use_trace]
        self.vp = vp[use_layer, use_trace]
        self.vs = vs[use_layer, use_trace]
        self.use_layer = []
        self.use_layer.append(0 if use_layer.start is None else use_layer.start)
        self.use_layer.append(self.vp.shape[0] if use_layer.stop is None else use_layer.stop)
        if vp.shape != vs.shape or vp.shape != rho.shape:
            raise ValueError("vp, vs rho 数据长度不一致，请检查代码")
        self.ntraces = 1 if len(self.vp.shape) == 1 else self.vp.shape[1]
        if self.ntraces != 1:
            self.use_trace = []
            self.use_trace.append(0 if use_trace.start is None else use_trace.start)
            self.use_trace.append(self.vp.shape[1] if use_trace.stop is None else use_trace.stop)
        else:
            self.use_trace=use_trace
        self.layers = self.vp.shape[0]
        # 低频数据
        nsmooth = 15
        self.vp_back = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, self.vp, axis=0)
        # vp_back = np.squeeze(scio.loadmat('Vpinit.mat')['vp_init'])
        self.vs_back = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, self.vs, axis=0)
        self.rho_back = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, self.rho, axis=0)

    def get_n(self):
        if self.vp != self.vs or self.vp != self.rho:
            raise ValueError("vp, vs rho 数据长度不一致，请检查代码")

    def __str__(self):
        if self.ntraces != 1:
            return f'训练数据为多道数据, 大小为 [layers={self.use_layer}, traces={self.use_trace}]'
        else:
            return f'训练数据为单道数据， 第{self.use_trace}道， 大小为[layers={self.use_layer},]'

    def show(self):
        def format_x_ticks(x, pos):
            return int(x + self.use_trace[0])

        def format_y_ticks(y, pos):
            return int(y + self.use_layer[0])

        if self.ntraces != 1:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='all')
            ax1.imshow(self.vp, aspect='auto')
            ax1.set_title('vp')
            ax1.set_xlabel('traces')
            ax1.set_ylabel('layers')
            x_formatter = FuncFormatter(format_x_ticks)
            ax1.xaxis.set_major_formatter(x_formatter)
            y_formatter = FuncFormatter(format_y_ticks)
            ax1.yaxis.set_major_formatter(y_formatter)
            ax2.imshow(self.vs, aspect='auto')
            ax2.set_title('vs')
            ax2.set_xlabel('traces')
            ax2.set_ylabel('layers')
            ax3.imshow(self.rho, aspect='auto')
            ax3.set_title('rho')
            ax3.set_xlabel('traces')
            ax3.set_ylabel('layers')
        else:
            pass
        plt.show()


class Real_Dataset():
    def __init__(self, vp_init, vs_init, rho_init, seismic_data, theta, use_trace, use_layer):
        """

        Args:
            vp_init: 二维时shape应为[layers, trace]
            vs_init: 二维时shape应为[layers, trace]
            rho_init: 二维时 shape应为[layers, trace]
            seismic_data: 二维时shape应为[layers, theta]; 三维时shape应为[trace, layers, theta];
            theta: 角度制的角度集
        """
        if vp_init.shape != vs_init.shape or vp_init.shape != rho_init.shape:
            raise ValueError("vp, vs rho 数据长度不一致，请检查代码")
        self.theta_rad = np.radians(theta)
        self.vp_init = vp_init[use_layer, use_trace]
        self.vs_init = vs_init[use_layer, use_trace]
        self.rho_init = rho_init[use_layer, use_trace]
        self.seismic_data = seismic_data[use_layer, use_trace, :]
        self.ntraces = 1 if len(self.vp_init.shape) == 1 else self.vp_init.shape[1]
        self.layers = vp_init.shape[0]

    def checkdata(self):  # TODO 当vp为一维时但shape为[layers, 1]，处理方法不同
        if self.ntraces == 1:
            if len(self.vp_init) != self.seismic_data.shape[0]:
                raise ValueError("单道：模型与数据层数不一致")
        else:
            if self.vp_init.shape[0] != self.seismic_data.shape[1]:
                raise ValueError("多道：模型与数据层数不一致")
            if self.vp_init.shape[-1] != self.seismic_data.shape[0]:
                raise ValueError("多道：模型与数据道数不一致")

class DataReader(metaclass=ABCMeta):
    @abstractmethod
    def read(self, datapath):
        pass


class Read_Marmousi_Imp(DataReader):
    def read(self, datapath):
        data = scio.loadmat(datapath)
        self.vp = data['Marmousi_Imp'] * 1000
        self.vs = self.vp * 0.7
        self.rho = 0.31 * self.vp ** 0.25 * 1000
        self.layers = self.vp.shape[0]
        self.traces = self.vp.shape[1]

    def __str__(self):
        return '读取Marmousi阻抗数据，根据经验公式得到弹性参数，\nvp数据大小 [layers=%d, traces=%d]' % (self.layers, self.traces)


class Read_Marmousi2(DataReader):
    def read(self, datapath):  # vp shape: [layers, traces]
        data = scio.loadmat(datapath)
        self.vp = data['vp']
        self.vs = data['vs']
        self.rho = data['den']
        self.layers = self.vp.shape[0]
        self.traces = self.vp.shape[1]

    def __str__(self):
        return '读取MarmousiII数据，\nvp数据大小 [layers=%d, traces=%d]' % (self.layers, self.traces)


class Settings(metaclass=ABCMeta):
    @abstractmethod
    def setup(self):
        pass


class Set_origin(Settings):
    def __init__(self, layers, dt0, wave_f, wave_n0, theta_list):
        """

        Args:
            layers: 采样点即层数
            dt0: 采样时间序列
            wave_f: 子波主频
            theta_list: 角度集 (可省略)
        """
        self.wave_f = wave_f
        self.dt0 = dt0
        self.t = (np.arange(layers + 1) * dt0)[1:]
        self.theta = theta_list
        self.wave_n0 = wave_n0
        self.layers = layers

    def __str__(self):
        return '使用原始方法生成子波，\n子波主频为%d, 长度为%d\n角度序列为%s' % (self.wave_f, self.layers, self.theta)

    def o_ricker(self, t, f0: float = 25):
        """

        Args:
            t: 时域抽样序列
            f0: 主频

        Returns:
            Ricker子波时域幅值

        """
        w = (1 - 2 * (np.pi * f0 * (t - 1 / f0)) ** 2) * np.exp(-((np.pi * f0 * (t - 1 / f0)) ** 2))  # 时移 1/f0??
        return w

    def o_covmtx(self, h):  # TODO
        """
        原始生成子波方法 origin method
        由子波生成矩阵，使该矩阵点乘代表原序列卷积

        Args:
            h: 所要变换的序列

        Returns:
            h_mtx: 可由点乘代表卷积的矩阵

        """
        npad = self.layers - 1
        h_mtx = np.vstack([np.pad(h, (i, npad - i), 'constant', constant_values=(0, 0)) for i in range(self.layers)]).T
        h_mtx = h_mtx[math.floor(1 / self.wave_f / self.dt0): math.floor(1 / self.wave_f / self.dt0) + self.layers, :]
        return h_mtx

    def setup(self):
        wav = self.o_ricker(self.t[: self.wave_n0], self.wave_f)
        self.wavemat = self.o_covmtx(wav)


class Set_Zps(Settings):
    def __init__(self, layers, dt0, wave_f, theta_list):
        """

        Args:
            layers: 采样点即层数
            dt0: 采样时间序列
            wave_f: 子波主频
            theta_list: 角度集
        """
        self.wave_f = wave_f
        self.t = np.arange(layers) * dt0
        self.theta = theta_list
        self.layers = layers

    def z_ricker(self, t, f0: float = 25):
        """

        Args:
            t: 时域抽样点
            f0: 主频

        Returns:
            Ricker子波时域幅值

        """

        t = np.concatenate((np.flipud(-t[1:]), t), axis=0)  # 将时域映射至整个实轴
        w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-((np.pi * f0 * t) ** 2))  # Ricker子波计算公式
        return w

    def z_covmtx(self, h, n):  # TODO 循环卷积？
        """
        得到卷积矩阵，
        即h_mtx*x等效于h卷积x
        Args:
            h: darray[n.]
            n: x序列长度

        Returns:
            h_mtx:

        """
        npad = n - 1
        h_mtx = np.vstack([np.pad(h, (i, npad - i), 'constant', constant_values=(0, 0)) for i in range(n)]).T
        h_mtx = h_mtx[len(h) // 2: -len(h) // 2 + 1, :]  # 掐头去尾，可不选
        return h_mtx

    def setup(self):
        wav = self.z_ricker(self.t[: self.layers // 2 + 1], self.wave_f)
        self.wavemat = self.z_covmtx(wav, self.layers)
