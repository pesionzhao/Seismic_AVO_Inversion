"""
@File :utils.py
@Author :Pesion
@Date :2023/9/18
@Desc : 
"""
import yaml
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import scipy.signal as signal
from Forward.Zoeppritz import Zoeppritz
from Forward.Zoeppritz_Complex import ZoeppritzComp
from util.DataSet import Set_origin
import math


def read_yaml(config_path, method=None):
    """

    Args:
        config_path: yaml文件路径
        method:

    Returns:

    Notes:
        读取yaml配置文件

    """
    with open(config_path, 'r', encoding="utf-8") as file:
        # data = file.read()
        cfg = yaml.safe_load(file)
    if method is not None:
        return cfg[method]
    else:
        return cfg


def soft_thresh(alpha, x):
    # return np.multiply(np.sign(x), np.maximum(np.abs(x) - alpha, 0))
    T = np.maximum(np.abs(x) - alpha, 0.0) * np.sign(x)
    # if sum(T)==0:
    #     T=x
    return T


def read_mat(mat_path: str, item: list, T=False, vmin=None, vmax=None, show=True):
    """

    Args:
        mat_path: 路径
        item: 需要查看的数据的key值
        T: 显示时数据是否转置--由于角道集数据shape:[traces,layers,theta],取角度维度进行查看时会转置
        vmin: imshow时固定vmin
        vmax: imshow时固定vmax
        show: 是否显示图片


    Notes:
        用于查看mat文件中的数据

    """
    data = scio.loadmat(mat_path)
    mat = {}
    for key, value in data.items():
        if key in item:
            if hasattr(value, 'shape'):
                if len(value) == 1 and value.shape[-1] == 1:
                    atrribute = 'value'
                else:
                    plt.figure()
                    if T:
                        value = value.T
                    mat.update({key: value})
                    if show:
                        if len(value.shape) > 2:
                            plt.imshow(value[:, :, 1].T, cmap='seismic', aspect='auto')
                        else:
                            plt.imshow(value, cmap='viridis', aspect='auto')
                    # plt.plot([999, 999], [0, 511], color="red", linewidth=2)
                    # plt.plot([2499, 2499], [0, 511], color="red", linewidth=2)
                    vmax = np.max(value)
                    vmin = np.min(value)
                    plt.title(key)
                    value = str(value.shape) + ' max:{:3f}, min{:3f}'.format(vmax, vmin)
                    atrribute = 'shape'
            else:
                atrribute = 'value'
            print(f'Key: {key}, {atrribute}: {value}')
        else:
            if hasattr(value, 'shape'):
                print(f'there is Key: {key}, shape {value.shape}')
            else:
                print(f'there is Key: {key}')
    return mat


def plot_result(pre, vp, vs, rho, vp_back, vs_back, rho_back):
    # 单道时
    if len(vp.shape) == 1:
        layers = len(vp)
        vp_cal = pre[:layers]
        vs_cal = pre[layers:2 * layers]
        rho_cal = pre[2 * layers:]
        # plt.figure()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
        ax1.plot(vp_cal, label='inv_vp')
        ax1.plot(vp[:], label='vp')
        ax1.plot(vp_back, label='vp_back')
        ax1.set_title('vp curve', loc='left')
        ax1.set_xlabel('layers')
        ax1.legend()

        ax2.plot(vs_cal, label='inv_vs')
        ax2.plot(vs[:], label='vs')
        ax2.plot(vs_back, label='vs_back')
        ax2.set_title('vs curve', loc='left')
        ax2.set_xlabel('layers')
        ax2.legend()

        ax3.plot(rho_cal, label='inv_rho')
        ax3.plot(rho[:], label='rho')
        ax3.plot(rho_back, label='rho_back')
        ax3.set_title('rho curve', loc='left')
        ax3.set_xlabel('layers')
        ax3.legend()
    # 多道时
    else:
        ntrace = vp.shape[-1]
        layers = vp.shape[0]
        vpnorm = plt.Normalize(vp.min(), vp.max()) # 用于统一标签与预测值色标
        vsnorm = plt.Normalize(vs.min(), vs.max())
        rhonorm = plt.Normalize(rho.min(), rho.max())
        vp_cal = pre[:layers]
        vs_cal = pre[layers:2 * layers]
        rho_cal = pre[2 * layers:]
        fig, axes = plt.subplots(2, 3)
        imvpcal = axes[0, 0].imshow(vp_cal, aspect='auto', norm=vpnorm)
        axes[0, 0].set_title('vp inv')
        fig.colorbar(imvpcal, ax=[axes[0, 0], axes[1, 0]])
        imvp = axes[1, 0].imshow(vp, aspect='auto', norm=vpnorm)
        axes[1, 0].set_title('vp')
        # fig.colorbar(imvp)

        imvscal = axes[0, 1].imshow(vs_cal, aspect='auto', norm=vsnorm)
        axes[0, 1].set_title('vs inv', )
        fig.colorbar(imvscal, ax=[axes[0, 1], axes[1, 1]])
        imvs = axes[1, 1].imshow(vs, aspect='auto', norm=vsnorm)
        axes[1, 1].set_title('vs')
        # fig.colorbar(imvs)

        imrhocal = axes[0, 2].imshow(rho_cal, aspect='auto', norm=rhonorm)
        axes[0, 2].set_title('rho inv')
        fig.colorbar(imrhocal, ax=[axes[0, 2], axes[1, 2]])
        imrho = axes[1, 2].imshow(rho, aspect='auto', norm=rhonorm)
        axes[1, 2].set_title('rho')
        # fig.colorbar(imrho)



def plot_single_trace(pre, vp, vs, rho, back):
    cut = 30
    layers = int(len(pre) / 3)
    vp_cal = pre[cut:layers]
    vs_cal = pre[layers + cut:2 * layers]
    rho_cal = pre[2 * layers + cut:]
    vp_back = back[cut:layers]
    vs_back = back[layers + cut:2 * layers]
    rho_back = back[2 * layers + cut:]
    # plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    ax1.plot(vp_cal, label='inv_vp')
    ax1.plot(vp[cut:], label='vp')
    ax1.plot(np.exp(vp_back), label='mean_vp')
    ax1.set_title('vp curve', loc='left')
    ax1.set_xlabel('layers')
    ax1.legend()

    ax2.plot(vs_cal, label='inv_vs')
    ax2.plot(vs[cut:], label='vs')
    ax2.plot(np.exp(vs_back), label='mean_vs')
    ax2.set_title('vs curve', loc='left')
    ax2.set_xlabel('layers')
    ax2.legend()

    ax3.plot(rho_cal, label='inv_rho')
    ax3.plot(rho[cut:], label='rho')
    ax3.plot(np.exp(rho_back), label='mean_rho')
    ax3.set_title('rho curve', loc='left')
    ax3.set_xlabel('layers')
    ax3.legend()
    plt.show()


def plot_multi_traces(pre, vp, vs, rho):
    """

    Args:
        pre: [3*layers, traces]
        vp: [layers, traces]
        vs: [layers, traces]
        rho: [layers, traces]

    Returns:

    """
    ntrace = vp.shape[-1]
    layers = vp.shape[0]
    vpnorm = plt.Normalize(vp.min(), vp.max())
    vsnorm = plt.Normalize(vs.min(), vs.max())
    rhonorm = plt.Normalize(rho.min(), rho.max())
    vp_cal = pre[:layers]
    vs_cal = pre[layers:2 * layers]
    rho_cal = pre[2 * layers:]
    # plt.figure()
    fig, axes = plt.subplots(2, 3)
    imvpcal = axes[0, 0].imshow(vp_cal, aspect='auto', norm=vpnorm)
    axes[0, 0].set_title('vp inv')
    fig.colorbar(imvpcal, ax=[axes[0, 0], axes[1, 0]])
    imvp = axes[1, 0].imshow(vp, aspect='auto', norm=vpnorm)
    axes[1, 0].set_title('vp')
    # fig.colorbar(imvp)

    imvscal = axes[0, 1].imshow(vs_cal, aspect='auto', norm=vsnorm)
    axes[0, 1].set_title('vs inv', )
    fig.colorbar(imvscal, ax=[axes[0, 1], axes[1, 1]])
    imvs = axes[1, 1].imshow(vs, aspect='auto', norm=vsnorm)
    axes[1, 1].set_title('vs')
    # fig.colorbar(imvs)

    imrhocal = axes[0, 2].imshow(rho_cal, aspect='auto', norm=rhonorm)
    axes[0, 2].set_title('rho inv')
    fig.colorbar(imrhocal, ax=[axes[0, 2], axes[1, 2]])
    imrho = axes[1, 2].imshow(rho, aspect='auto', norm=rhonorm)
    axes[1, 2].set_title('rho')
    # fig.colorbar(imrho)
    plt.show()


def augdata(vp, vs, rho, index: list, addtrace=9):
    """

    Args:
        addtrace: 需要扩充的道数
        vp: 原始数据[layers, traces]
        vs:
        rho:
        index: 抽取的地震道索引 类型为列表[]

    Returns:
        加噪后的数据:vp_aug, vs_aug, rho_aug

    Notes:
        由于测井数据较少,需要通过数据增广来增大样本数,这里采用加噪的方式扩充, 通过原始数据的index索引得到模拟的测井数据

    """
    chose_vp = []
    chose_vs = []
    chose_rho = []
    for i in index:
        chose_vp.append(vp[:, i - 1])
        chose_vs.append(vs[:, i - 1])
        chose_rho.append(rho[:, i - 1])

    fig1, ax_vp = plt.subplots(1, len(index), sharey='all')
    ax_vp[0].set_ylabel('vp')
    fig2, ax_vs = plt.subplots(1, len(index), sharey='all')
    ax_vs[0].set_ylabel('vs')
    fig3, ax_rho = plt.subplots(1, len(index), sharey='all')
    ax_rho[0].set_ylabel('rho')
    fig1.suptitle(f'vp label')
    fig2.suptitle(f'vs label')
    fig3.suptitle(f'rho label')
    for i in range(len(index)):
        ax_vp[i].plot(chose_vp[i])
        ax_vp[i].set_title(f'trace {index[i]}')
        ax_vp[i].set_xlabel('layers')
        ax_vs[i].plot(chose_vs[i])
        ax_vs[i].set_title(f'trace {index[i]}')
        ax_vs[i].set_xlabel('layers')
        ax_rho[i].plot(chose_rho[i])
        ax_rho[i].set_title(f'trace {index[i]}')
        ax_rho[i].set_xlabel('layers')

    noise = np.random.normal(0, 0.01, (vp.shape[0], addtrace))
    aug_vp = None
    aug_vs = None
    aug_rho = None
    for i in range(len(index)):
        if i == 0:
            aug_vp = np.concatenate([chose_vp[i][:, None], chose_vp[i][:, None] + noise], axis=1)
            aug_vs = np.concatenate([chose_vs[i][:, None], noise + chose_vs[i][:, None]], axis=1)
            aug_rho = np.concatenate([chose_rho[i][:, None], noise + chose_rho[i][:, None]], axis=1)
        else:
            aug_vp = np.concatenate([aug_vp, chose_vp[i][:, None], chose_vp[i][:, None] + noise], axis=1)
            aug_vs = np.concatenate([aug_vs, chose_vs[i][:, None], noise + chose_vs[i][:, None]], axis=1)
            aug_rho = np.concatenate([aug_rho, chose_rho[i][:, None], noise + chose_rho[i][:, None]], axis=1)
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(aug_vp)
    ax[0].set_title('aug_vp')
    ax[1].plot(aug_vs)
    ax[1].set_title('aug_vs')
    ax[2].plot(aug_rho)
    ax[2].set_title('aug_rho')
    plt.show()

    return aug_vp, aug_vs, aug_rho


def back_model(vp, vs, rho, f0=5):
    """

    Args:
        vp:
        vs:
        rho:
        f0: 截止频率

    Returns:
        低频初始模型 [vp_back, vs_back, rho_back]

    """
    N = 2
    dt = 0.002
    Wn = 2 * f0 / (1 / (dt))  # 归一化截止频率, 计算公式Wn=2*截止频率/采样频率。
    # Wn = 0.003  # 归一化截止频率, 计算公式Wn=2*截止频率/采样频率。
    b, a = signal.butter(N, Wn, btype='low')
    vp_back = signal.filtfilt(b, a, vp, axis=0)
    vs_back = signal.filtfilt(b, a, vs, axis=0)
    rho_back = signal.filtfilt(b, a, rho, axis=0)
    fig, ax = plt.subplots(1, 3)
    if len(vp_back.shape) == 2:
        ax[0].imshow(vp_back, aspect='auto')
        ax[0].set_title('vp_back')
        ax[1].imshow(vs_back, aspect='auto')
        ax[1].set_title('vs_back')
        ax[2].imshow(rho_back, aspect='auto')
        ax[2].set_title('rho_back')
    elif len(vp_back.shape) == 1:
        ax[0].plot(vp_back)
        ax[0].set_title('vp_back')
        ax[1].plot(vs_back)
        ax[1].set_title('vs_back')
        ax[2].plot(rho_back)
        ax[2].set_title('rho_back')
    return vp_back, vs_back, rho_back


def v2seis(vp, vs, rho, wavemat, theta1, theta2, theta3):
    """

    Args:
        vp: shape: [layers, traces]
        vs:
        rho:
        wavemat: shape: [layers, layers]
        theta1: 角度制!!
        theta2:
        theta3:

    Notes:
        正演角道集数据

    Returns:

    """
    # changeable parameter
    forward = ZoeppritzComp(vp.shape[-1], vp.shape[0])

    seismic1 = forward.forward(vp, vs, rho, np.radians(theta1), wavemat)
    seismic2 = forward.forward(vp, vs, rho, np.radians(theta2), wavemat)
    seismic3 = forward.forward(vp, vs, rho, np.radians(theta3), wavemat)
    return seismic1, seismic2, seismic3


def generate_ricker(layers, f0, dt0):
    t = (np.arange(layers + 1) * dt0)[1:]
    w = (1 - 2 * (np.pi * f0 * (t - 1 / f0)) ** 2) * np.exp(-((np.pi * f0 * (t - 1 / f0)) ** 2))  # w 为子波序列
    npad = layers - 1  # 卷积起始至少一个元素,其他都为零,所以pad最大layers-1
    h_mtx = np.vstack([np.pad(w, (i, npad - i), 'constant', constant_values=(0, 0)) for i in
                       range(layers)]).T  # 大小为[2*layers-1, layers]
    h_mtx = h_mtx[math.floor(1 / f0 / dt0): math.floor(1 / f0 / dt0) + layers, :]
    return h_mtx


def generate_dataset(vp, vs, rho, cutoff_f=5, name='Red'):
    """

    Args:
        vp:
        vs:
        rho:
        cutoff_f: 截止频率
        name: 生成文件名字前缀

    Notes:
        生成数据集, 按照name得到name_train_dataset.mat和name_test_dataset.mat

    """
    train_data = generate_train_dataset(vp, vs, rho, cutoff_f)
    scio.savemat(name + '_train_dataset.mat', train_data)
    test_data = generate_test_dataset(vp, vs, rho, cutoff_f)
    scio.savemat(name + '_test_dataset.mat', test_data)


def update_back(train_data, test_data, f=1):
    """

    Args:
        train_data: 训练集mat文件
        test_data: 测试集mat文件
        f: 截止频率
    Notes:
        根据不同的截止频率生成不同的初始模型

    """
    vp_aug, vs_aug, rho_aug = train_data['vp_aug'], train_data['vs_aug'], train_data['rho_aug']
    train_data['vp_back'], train_data['vs_back'], train_data['rho_back'] = back_model(vp_aug, vs_aug, rho_aug, f)
    scio.savemat(f'train_dataset_f={f}.mat', train_data)

    vp, vs, rho = test_data['vp'], test_data['vs'], test_data['rho']
    test_data['vp_back'], test_data['vs_back'], test_data['rho_back'] = back_model(vp, vs, rho, f)
    scio.savemat(f'test_dataset_f={f}.mat', test_data)
    return 0


def generate_train_dataset(vp, vs, rho, cutoff_f):
    """

    Args:
        vp: 真实测井数据
        vs:
        rho:
        cutoff_f: 生成低频模型的截止频率
    Note:
        使用真实测井数据生成训练集
    Returns:
        mat数据, \\ 其中包含{背景数据_back,测井数据_aug,叠前角道集seis_s/m/l,子波矩阵wavemat,角度theta}

    """
    # parameter
    # TODO 主频, 采样间隔, 入射角度, 抽取的道数作为可变参数由外部控制,目前不用
    f0 = 30
    dt0 = 0.002
    theta1 = np.arange(1, 16)
    theta2 = np.arange(16, 31)
    theta3 = np.arange(31, 46)
    vp, vs, rho = augdata(vp, vs, rho, [100, 600])

    vp_back, vs_back, rho_back = back_model(vp, vs, rho, cutoff_f)
    wavemat = generate_ricker(vp.shape[0], f0, dt0)
    seismic1, seismic2, seismic3 = v2seis(vp, vs, rho, wavemat, theta1, theta2, theta3)
    dataset = {}
    dataset.update({'vp_back': vp_back})
    dataset.update({'vs_back': vs_back})
    dataset.update({'rho_back': rho_back})
    dataset.update({'vp_aug': vp})
    dataset.update({'vs_aug': vs})
    dataset.update({'rho_aug': rho})
    dataset.update({'seis_s': seismic1})
    dataset.update({'seis_m': seismic2})
    dataset.update({'seis_l': seismic3})
    dataset.update({'wavemat': wavemat})
    dataset.update({'theta': [theta1, theta2, theta3]})
    return dataset


def generate_test_dataset(vp, vs, rho, cutoff_f):
    """

    Args:
        vp: 测试标签
        vs:
        rho:
        cutoff_f: 生成低频模型的截止频率

    Notes:
        使用标签数据生成测试数据用于评估(仅针对合成数据也就是有标签数据,由于真实数据是无标签的,故无法通过预测结果评估)

    Returns:
        mat数据,\\其中包含{背景数据_back,标签数据_label,叠前角道集seis_s/m/l,子波矩阵wavemat,角度theta}

    """
    # parameter

    wavemat = generate_ricker(vp.shape[0], 30, 0.002)
    theta1 = np.arange(1, 16)
    theta2 = np.arange(16, 31)
    theta3 = np.arange(31, 46)

    vp_back, vs_back, rho_back = back_model(vp, vs, rho, cutoff_f)
    seismic1, seismic2, seismic3 = v2seis(vp, vs, rho, wavemat, theta1, theta2, theta3)
    dataset = {}
    dataset.update({'vp': vp})
    dataset.update({'vs': vs})
    dataset.update({'rho': rho})
    dataset.update({'vp_back': vp_back})
    dataset.update({'vs_back': vs_back})
    dataset.update({'rho_back': rho_back})
    dataset.update({'seis_s': seismic1})
    dataset.update({'seis_m': seismic2})
    dataset.update({'seis_l': seismic3})
    dataset.update({'wavemat': wavemat})
    dataset.update({'theta': [theta1, theta2, theta3]})
    return dataset


def img2line(vp, vs, rho):
    fig1 = plt.figure(22)
    plt.title('vp')
    plt.legend()
    fig2 = plt.figure(33)
    plt.title('vs')
    plt.legend()
    fig3 = plt.figure(44)
    plt.title('rho')
    plt.legend()
    for i in range(2):
        plt.figure(fig1.number)
        plt.plot(vp[i + 3], label=f'{i}')
        plt.figure(fig2.number)
        plt.plot(vs[i + 3], label=f'{i}')
        plt.figure(fig3.number)
        plt.plot(rho[i + 3], label=f'{i}')


def extract_seis(vp, vs, rho, layer_ratio, trace_ratio):
    """

    Args:
        vp:
        vs:
        rho:
        layer_ratio:
        trace_ratio:
    Notes:
        从原始数据中抽取一部分用作实验，可以理解为采样/压缩，vp等维度必须为[layers, traces]
    Returns:
        采样后的速度和密度
    """
    layers = vp.shape[0]
    traces = vp.shape[1]
    layer_index = np.arange(0, layers, 1 / layer_ratio).astype(int)
    trace_index = np.arange(0, traces, 1 / trace_ratio).astype(int)
    vp = vp[np.ix_(layer_index, trace_index)]
    vs = vs[np.ix_(layer_index, trace_index)]
    rho = rho[np.ix_(layer_index, trace_index)]
    layers = vp.shape[0]
    layers = 2**int(math.log2(layers))

    return vp[:layers], vs[:layers], rho[:layers]
