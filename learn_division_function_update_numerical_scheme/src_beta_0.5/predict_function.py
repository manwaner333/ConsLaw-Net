import numpy as np
from numpy import *
import torch
from torch.autograd import Variable
from aTEAM.nn.modules.Interpolation import LagrangeInterpFixInputs, LagrangeInterp
import torchvision
from torchviz import make_dot
import expr
from torch.autograd import grad
__all__ = ['VariantCoeLinear1dPredict']


class VariantCoeLinear1dPredict(torch.nn.Module):
    def __init__(self, T, N, X, batch_size, u0, dt, time_steps, dx, M, max_f_prime, u_fixed, layer, device, is_train=True):
        super(VariantCoeLinear1dPredict, self).__init__()
        coe_num = 1  # the number of coefficient
        self.coe_num = coe_num
        self.T = T
        self.N = N  # The number of grid cell
        self.X = X
        self.batch_size = batch_size
        self.allchannels = ['u']
        self.layer = layer
        self.channel_num = 1
        self.hidden_layers = 2
        polys = []
        for k in range(self.channel_num):
            self.add_module('poly'+str(k), expr.poly(self.hidden_layers, channel_num=len(self.allchannels), channel_names=self.allchannels))
            polys.append(self.__getattr__('poly'+str(k)))
        self.polys = tuple(polys)
        self.register_buffer('u0', u0)
        self.register_buffer('u_fixed', u_fixed)
        self.register_buffer('dt', torch.DoubleTensor(1).fill_(dt))
        self.register_buffer('dx', torch.DoubleTensor(1).fill_(dx))
        self.register_buffer('M', torch.DoubleTensor(1).fill_(M))
        self.register_buffer('max_f_prime', torch.DoubleTensor(1).fill_(max_f_prime))
        self.register_buffer('time_steps', torch.DoubleTensor(1).fill_(time_steps))
        self.device = device
        self.is_train = is_train

    @property
    def coes(self):
        for i in range(self.coe_num):
            yield self.__getattr__('coe'+str(i))
    @property
    def xy(self):
        return Variable(next(self.coes).inputs)
    @xy.setter
    def xy(self, v):
        for fitter in self.coes:
            fitter.inputs = v

    def coe_params(self):
        parameters = []
        for poly in self.polys:
            parameters += list(poly.parameters())
        return parameters

    def f_predict(self, u):
        u = u.unsqueeze(1)
        Uadd = list(poly(u.permute(0, 2, 1)) for poly in self.polys)
        uadd = torch.cat(Uadd, dim=1)
        return uadd

    def f_real(self, u):
        # molecular = (-0.5261977708897227)*u**2+(-0.42022981191563563)*u+(0.16331364238492996)*1+(-0.004928467833697386)*u**3+(1.7494189444438742e-05)*u**4
        # denominator = (-1.819039406865834)*u**2+(1.4471178621267011)*u+(-0.6862305238126161)*1+(-0.017404071489114364)*u**3+(6.177784536877158e-05)*u**4
        # molecular = (-0.8856238306912274)*u+(0.45938973625587726)*1+(-0.09946683091133793)*u**2+(-0.0009517213279790672)*u**3+(3.0001620679894885e-06)*u**4
        # denominator = (-2.014375886749311)*u**2+(1.3288025912366521)*u+(-0.7034363986145952)*1+(-0.022061358836315586)*u**3+(6.954520194431801e-05)*u**4
        # molecular = (-1.183961236879285)*u+(0.6496037744733122)*1+(0.5382167173224807)*u**2+(0.00035533795165860273)*u**3
        # denominator = (-2.317863264018614)*u**2+(1.2777256645212387)*u+(-0.6494546138917048)*1+(-0.0015282106523802684)*u**3
        molecular = (-1.1652052887708202)*u+(0.5864813977648344)*1+(0.4012576646892352)*u**2+(3.3237370892124386e-05)*u**3
        denominator = (-2.0829050389978403)*u**2+(1.3475365242369104)*u+(-0.6698385882971303)*1+(-0.00017249228191796602)*u**3
        return molecular/denominator

    def f_half(self, u):
        if self.is_train:
            f = 0.05 * self.f_predict(u)
        else:
            f = 0.05 * self.f_real(u)
        f_half = torch.empty((self.batch_size, self.N - 1), requires_grad=False).to(self.device)
        for index in range(self.N - 1):
            f_half[:, index] = 0.5 * (f[:, index] + f[:, index + 1]) - 0.5 * self.M * (u[:, index + 1] - u[:, index])
        return f_half

    # 没有系数
    def a(self):
        x = []
        for index in range(self.N):
            if index == 0:
                x.append(self.dx * 0.5)
            else:
                x.append(self.dx * 0.5 + index * self.dx)
        # true parameters: 0.5 * x
        res = torch.empty((1, self.N), requires_grad=False).to(self.device)
        for index in range(self.N):
            res[:, index] = 1.0
        return res.repeat(self.batch_size, 1).double()

    def df_du(self, u):
        if self.is_train:
            f = 0.05 * self.f_predict(u)
        else:
            f = 0.05 * self.f_real(u)
        # 计算目前f(u)下面的导数
        dfdu = grad(f, u, grad_outputs=torch.ones_like(f), create_graph=False)[0]
        max_f_prime = torch.max(torch.abs(dfdu)).item()
        max_f_prime = round(max_f_prime+0.01, 2)
        return max_f_prime

    def update(self):
        # 计算目前状况下f(u)导数的最大值
        self.u_fixed.requires_grad = True
        max_f_prime = self.df_du(self.u_fixed)
        self.u_fixed.requires_grad = False
        # 如果　max_f_prime　比　self.max_f_prime大，　需要更新self.dt, self.M, self.max_f_prime 和 self.time_step
        # if self.max_f_prime.item() < max_f_prime and max_f_prime < 1:
        if 0.04 < max_f_prime  and max_f_prime < 3:
            print("adjust max_f_prime")
            # the new version of numerical scheme
            dt_a = 0.75 * self.dx.item()/(max_f_prime + 0.0001)
            n_time = self.T/dt_a
            n_time = int(round(n_time+1, 0))
            dt = self.T/n_time
            M = max_f_prime
            qingli = 4
            self.max_f_prime = torch.DoubleTensor(1).fill_(max_f_prime).to(self.device)
            self.dt = torch.DoubleTensor(1).fill_(dt).to(self.device)
            self.time_steps = torch.IntTensor(1).fill_(n_time).to(self.device)
            self.M = torch.DoubleTensor(1).fill_(M).to(self.device)
            print("\033[34mmax_f_prime %.6f, dt %.6f, time_steps %.6f, m %.6f,\033[0m" % (self.max_f_prime, self.dt, self.time_steps, self.M))


    # 初始值传进来， 经过这个forward, stepnum步的值都会被计算出来， 注意此处的stepnum和self.time_steps是不一样的
    def forward(self, init, stepnum):
        # self.update()
        # init.requires_grad = False
        u_old = init
        dt = self.dt
        dx = self.dx
        coefficient = self.a()
        trajectories = torch.empty((stepnum, self.batch_size, self.N), requires_grad=False, device=self.device)
        trajectories[0, :, :] = u_old
        for i in range(1, stepnum):
            f_half = self.f_half(u_old)
            u = torch.empty((self.batch_size, self.N), requires_grad=False).to(self.device)
            for j in range(1, self.N - 1):
                u[:, j] = u_old[:, j] - coefficient[:, j] * (dt/dx) * (f_half[:, j] - f_half[:, j-1])
            u[:, 0] = u[:, 1]
            u[:, self.N - 1] = u[:, self.N - 2]
            u_old = u
            trajectories[i, :] = u_old
        return trajectories


def generate_predict_data(save_file):
    device = 'cpu'
    T = 40
    X = 10
    dt = 0.08
    dx = 0.025
    N = 400
    M = 0.1000
    time_steps = 200
    max_f_prime = -0.03
    batch_size = 1
    # # u_0
    # u_0_np = np.zeros((1, N), dtype=float)
    # u_0_np[:1, 160:240] = 0.8  # N = 400
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    # 加入噪声之后的
    u_0_file = 'data/' + experiment_name + '_u0' + '.npy'
    u_0 = torch.from_numpy(np.load(u_0_file))
    u_0 = u_0.to(device)
    # 引入 u_fixed, 用来计算max_f_prime
    du = 1.2/52
    u_fixed_0 = -0.1+0.5*du
    u_fixed_np = np.zeros((1, 52), dtype=float)
    u_fixed_np[:1, 0] = u_fixed_0
    for i in range(1, 52):
        u_fixed_np[:1, i] = u_fixed_0 + i * du
    u_fixed = torch.from_numpy(u_fixed_np)
    u_fixed = u_fixed.to(device)
    # model
    linpdelearner = VariantCoeLinear1dPredict(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
                                       , dx=dx, M=M, max_f_prime=max_f_prime, u_fixed=u_fixed, layer=20, device=device, is_train=False)
    linpdelearner.update()
    # 预测值
    U = linpdelearner(linpdelearner.u0, linpdelearner.time_steps)
    print("U")
    print(U.shape)
    print("u_0")
    print(u_0.shape)
    print("u_fixed")
    print(u_fixed.shape)
    np.save(save_file, U.detach().to('cpu'))



def generate_predict_data_cell_1600(save_file):
    device = 'cpu'
    T = 40.0
    X = 10
    dt = 0.023529
    dx = 10/1600   # 10/200   # 10/1600  # 2   # 0.05
    N = 1600  # 200  # 200  # 1600   # 200
    M = 0.1000
    time_steps = 200
    max_f_prime = -0.03
    batch_size = 1
    # # u_0
    # u_0_np = np.zeros((1, N), dtype=float)
    # u_0_np[:1, 640:960] = 0.8  # N = 400
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    # 加入噪声之后的
    u_0_file = 'data/' + experiment_name + '_u0' + '.npy'
    u_0 = torch.from_numpy(np.load(u_0_file))
    u_0 = u_0.to(device)
    # 引入 u_fixed, 用来计算max_f_prime
    du = 1.2/52
    u_fixed_0 = -0.1+0.5*du
    u_fixed_np = np.zeros((1, 52), dtype=float)
    u_fixed_np[:1, 0] = u_fixed_0
    for i in range(1, 52):
        u_fixed_np[:1, i] = u_fixed_0 + i * du
    u_fixed = torch.from_numpy(u_fixed_np)
    u_fixed = u_fixed.to(device)
    # model
    linpdelearner = VariantCoeLinear1dPredict(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
                                              , dx=dx, M=M, max_f_prime=max_f_prime, u_fixed=u_fixed, layer=20, device=device, is_train=False)
    linpdelearner.update()
    # 预测值
    U = linpdelearner(linpdelearner.u0, linpdelearner.time_steps)
    print("U")
    print(U.shape)
    print("u_0")
    print(u_0.shape)
    print("u_fixed")
    print(u_fixed.shape)
    np.save(save_file, U.detach().to('cpu'))




if __name__ == "__main__":
    N = 400
    # experiment_name = 'N_400_example_1_dt_0.6_layer_20_beta_0.5_noise_0.05'
    # experiment_name = 'N_400_example_1_dt_0.6_layer_25_beta_0.5'
    experiment_name = 'N_400_example_1_dt_0.6_layer_25_beta_0.5_noise_0.03'
    predict_data_file = 'data/' + 'predict_' + experiment_name + '_U' + '.npy'
    generate_predict_data(predict_data_file)

    # N = 1600
    # experiment_name = 'N_1600_example_1_dt_0.6_layer_20_beta_0.5'
    # real_data_file = 'data/' + experiment_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + experiment_name + '_U' + '.npy'
    # generate_predict_data_cell_1600(predict_data_file)
    # generate_predict_data_cell_1600(real_data_file)





