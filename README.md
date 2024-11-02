## 叠前地震数据三参数反演

工厂设计模式实现叠前三参数的线性反演与非线性反演，此仓库为解决优化非线性或线性方程 $\min\limits_x||g(x)-b||+\lambda(x)$
的框架, $g(x)$可以是任意与 $x$ 相关的函数

### 文件树

```
├─config  # 配置文件
├─Forward　# 正演方法以及梯度求解
├─Regularization # 正则化方法
├─Solver　＃　优化策略
├─util　#　其他
```

### 配置文件格式与基类注释

数据类型

vp/vs/rho的shae均为`[layers, traces]` 或 `[layers, ]`

seismic_data的shape为`[traces, layers, theta]` 或 `[layers, theta]`

#### Forward

博客网址 [反演基础](https://pesionzhao.github.io/SeismicInversion/Util/#_7)

`Forward/`文件夹为正演模型, 也就是$g(x)$的实现, 需要规定$g(x)$的以及$g(x)'$的计算方式, 目前已经实现了Zoeppritz非线性正演模型,
Aki-Richards非线性正演模型, 以及简化的Aki-Richards线性正演模型.

以下是Forward的基类, 如果想指定自己的正演模型,需要重写`forward()`: $g(x)$的计算方法, `jacobian`: $g(x)$的计算方法
以及`showresult()`显示数据方法

```python
class BasicForward:
    def forward(self, vp, vs, rho, theta, wavemat):
        """
        通过不同的正演方法获取反射系数
  
        Args:
            vp: 横波序列
            vs: 纵波序列
            rho: 密度
            theta: 入射角度集
            wavemat: 子波矩阵
  
        Returns:
            反射系数 rpp
  
        """
    def showresult(self, dt0, trace, theta):
        """
        显示正演后的地震数据

        Args:
            dt0: 采样间隔
            trace: 展示地震数据的道数，单道默认为0
            theta: 横坐标角度制的theta


        """
    def jacobian(self, vp, vs, rho, theta, wav_mtx):
        """
        Calculate Jacobian matrix numerically.
        J_ij = d(r_i)/d(x_j)
        """
```

#### Solver

博客网址: [优化方法笔记](https://pesionzhao.github.io/SeismicInversion/Optimization/#_2)

`Solver`文件夹为优化器, 对于可以将反问题写为如下形式的问题

$$
\min\limits_x||g(x)-b||^2_2
$$

可以使用此函数包进行迭代优化, 目前已经完成了梯度下降`GD_solver()`,高斯牛顿法`GN_solver()`, 列文伯格-马夸特方法(Levenberg–Marquardt)
`LM_solver()`, **Split Bergman方法`SB_solver()`还未完全写好**, 用户如想指定自己的优化器,需要继承`Solver`基类, 并重写`one_step()`方法
详细参数如下:

```python
class Solver:
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
          
            [G    ]     [d      ]
          
            |     | m = |       |
          
            [Regop]     [Regdata]

        Returns:
            一次迭代输出

        """
```

#### Regularization

如果优化问题有约束项$\lambda(x)$, 也就是对于$\min\limits_x||g(x)-b||+\lambda(x)$的问题, 则需要指定$\lambda(x)$,
目前只实现了简单的一范数$|x|_1$ (通过IRLS求解) 和二范数$||x||^2_2$约束

```python
class Regularization:
    def set_up(self, cfg, layers, ntraces):
        """

        Args:
            cfg: 配置文件mat,用于定义正则化超参数
            layers:
            ntraces:
        Notes:
            根据配置文件初始化正则化项

        """
      
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
```

在完成以上操作后,可以通过yaml文件构建优化器, 示例为 `config/MyCfg.yaml`.

```yaml
# 通过yaml文件进行建造者的实例化, 
# 用户可按如下方式指定自己的建造者实例,指定优化方法solver和正则化项reg即可

## 使用LM方法优化带有一范数约束的优化问题
LM_IRLS:
  solver:
    name: 'LM'
    iters: 100
    lamb: 0.001
    step_size: 6 # 步长
  reg:
    name: 'IRLS'
    alpha: 0.01
    eps: 0.1
```

正演的数据路径和超参数同样由config文件进行配置, 示例为 `config/prestack.yaml`

```yaml
datapath: # 数据路径
### 用于生成子波
dt0: # 采样时间
wave_f: # 主频
wave_n0: # 采样点

### 数据道抽取
use_trace: 100 # 如果为int类型即单道, 若为列表,格式为slice [begin,end,step]
use_layer: [100,300,null] #同上
theta: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 角度
```

### 实例化工厂

在指定好正演模型, 优化策略, 正则化项后, 可以进行实例化, 用于整个反演的流程, 在文件`Builder.py`下有示例, 初始化可以自己指定, 如下

```python
class SimulateBuilder(Builder):
    def __init__(self, dataset: Train_DataSet, settings: Settings, forwardmodel: ForwardModel, solver: Solver,
                 reg: Regularization, obs=None):
        self.dataset = dataset
        self.settings = settings
        self.forwardmodel = forwardmodel
        self.reg = reg
        self.solver = solver
        if obs is None:
            self.obs = self.forwardmodel.forward(self.dataset.vp, self.dataset.vs, self.dataset.rho, self.dataset.theta_rad,
                                                 self.settings.wavemat)
        else:
            self.obs = obs
        # print(dataset)
        print(forwardmodel)
        print(solver)
        print(reg)
```

此项目还存在很多不足之处, 还请大家多多包含, 仅供教学使用, 在处理大规模矩阵时, 会有爆内存的问题, 请大家注意

如果大家觉得这个项目不错,想一起发展的话也欢迎PR

#### TODO

- [ ]  一范数的优化
- [ ]  Split-Bregman方法未完成

欢迎访问我的博客和我交流 https://pesionzhao.github.io/

邮箱： pesionzhao@163.com
