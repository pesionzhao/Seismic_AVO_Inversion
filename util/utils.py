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


def read_yaml(config_path, method=None):
    with open(config_path, 'r') as file:
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


def read_mat(mat_path, T=False, vmin=None, vmax=None):
    data = scio.loadmat(mat_path)
    for key, value in data.items():
        if hasattr(value, 'shape'):
            if len(value) == 1 and value.shape[-1] == 1:
                atrribute = 'value'
            else:
                plt.figure()
                if T:
                    value = value.T
                plt.imshow(value, cmap='viridis',aspect='auto')
                vmax = np.max(value)
                vmin = np.min(value)
                plt.title(key)
                value = str(value.shape) + 'max:{:3f}, min{:3f}'.format(vmax, vmin)
                atrribute = 'shape'
        else:
            atrribute = 'value'

        print(f'Key: {key}, {atrribute}: {value}')


def plot_result(vp, vs, rho):
    plt.subplots(311)
    plt.imshow(vp)
    plt.subplots(312)
    plt.imshow(vs)
    plt.subplots(313)
    plt.imshow(rho)


def plot_single_trace(pre, vp, vs, rho):
    layers = int(len(pre) / 3)
    vp_cal = pre[:layers]
    vs_cal = pre[layers:2 * layers]
    rho_cal = pre[2 * layers:]
    # plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(vp_cal)
    ax1.plot(vp)

    ax2.plot(vs_cal)
    ax2.plot(vs)

    ax3.plot(rho_cal)
    ax3.plot(rho)
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
    vp_cal = pre[:layers]
    vs_cal = pre[layers:2 * layers]
    rho_cal = pre[2 * layers:]
    # plt.figure()
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].imshow(vp_cal, vmin=1510, vmax=1800)
    axes[0, 0].set_title('vp inv')
    axes[1, 0].imshow(vp)
    axes[1, 0].set_title('vp')

    axes[0, 1].imshow(vs_cal, vmin=310, vmax=510)
    axes[0, 1].set_title('vs inv', )
    axes[1, 1].imshow(vs)
    axes[1, 1].set_title('vs')

    axes[0, 2].imshow(rho_cal, vmin=1950, vmax=2050)
    axes[0, 2].set_title('rho inv')
    axes[1, 2].imshow(rho)
    axes[1, 2].set_title('rho')
    plt.show()
