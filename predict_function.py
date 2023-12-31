from numpy import *
import torch
from torch.autograd import Variable
import expr
from torch.autograd import grad
import numpy as np
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
        self.hidden_layers = 3
        polys = []
        for k in range(self.channel_num):
            self.add_module('poly' + str(k), expr.poly(self.hidden_layers, channel_num=len(self.allchannels), channel_names=self.allchannels))
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
        f = (1.5733140772285443)*u+(-0.8676433255901695)*u**3+(0.4623734302522332)*1+(0.23570323139025814)*u**4+(0.06594985868545625)*u**2+(-0.005876625881704264)*u**5+(5.1464175544821254e-05)*u**6
        return f

    def f_half(self, u):
        if self.is_train:
            f = 1.0 * self.f_predict(u)
        else:
            f = 1.0 * self.f_real(u)
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
            f = 1.0 * self.f_predict(u)
        else:
            f = 1.0 * self.f_real(u)
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
        dt_a = 0.75 * self.dx.item()/(max_f_prime + 0.0001)
        n_time = self.T/dt_a
        n_time = int(round(n_time+1, 0))
        dt = self.T/n_time
        M = max_f_prime
        self.max_f_prime = torch.DoubleTensor(1).fill_(max_f_prime).to(self.device)
        self.dt = torch.DoubleTensor(1).fill_(dt).to(self.device)
        self.time_steps = torch.IntTensor(1).fill_(n_time).to(self.device)
        self.M = torch.DoubleTensor(1).fill_(M).to(self.device)
        print("\033[34mmax_f_prime %.6f, dt %.6f, time_steps %.6f, m %.6f,\033[0m" % (self.max_f_prime, self.dt, self.time_steps, self.M))


    def forward(self, init, stepnum):
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

def generate_predict_data_cell_200(save_file):
    device = 'cpu'
    T = 2.0
    X = 10
    N = 200  # 200  # 200  # 1600   # 200
    dx = X/N  # 10/200   # 10/1600  # 2   # 0.05
    dt = 0.023529
    M = 1.580000
    time_steps = 85
    max_f_prime = 1.580000
    # 用来做训练的
    batch_size = 3
    # u_0
    u_0_np = np.zeros((1, N), dtype=float)
    u_0_np[:1, 0:60] = 1.0
    u_0 = torch.from_numpy(u_0_np)
    u_0 = u_0.to(device)
    # 引入 u_fixed, 用来计算max_f_prime
    du = 1.2/502
    u_fixed_0 = -0.1+0.5*du
    u_fixed_np = np.zeros((1, 502), dtype=float)
    u_fixed_np[:1, 0] = u_fixed_0
    for i in range(1, 502):
        u_fixed_np[:1, i] = u_fixed_0 + i * du
    u_fixed = torch.from_numpy(u_fixed_np)
    u_fixed = u_fixed.to(device)
    # model
    linpdelearner = VariantCoeLinear1dPredict(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
                                              , dx=dx, M=M, max_f_prime=max_f_prime, u_fixed=u_fixed, layer=20, device=device, is_train=False)
    # 预测值
    linpdelearner.update()
    U = linpdelearner(linpdelearner.u0, linpdelearner.time_steps)
    print("U")
    print(U.shape)
    print("u_0")
    print(u_0.shape)
    print("u_fixed")
    print(u_fixed.shape)
    np.save(save_file, U.detach().to('cpu'))


if __name__ == "__main__":
    experiment_name = 'N_200_example_1_dt_0.1_layer_10_beta_10'
    real_data_file = 'data/' + experiment_name + '.npy'
    predict_data_file = 'data/' + 'predict_' + experiment_name + '.npy'
    generate_predict_data_cell_200(predict_data_file)





