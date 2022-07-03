import numpy as np
from numpy import *
import torch
from torch.autograd import Variable
from aTEAM.nn.modules.Interpolation import LagrangeInterpFixInputs, LagrangeInterp
import torchvision
# from torchviz import make_dot
import expr
from torch.autograd import grad
__all__ = ['VariantCoeLinear1d']


class VariantCoeLinear1d(torch.nn.Module):
    def __init__(self, T, N, X, batch_size, u0, dt, time_steps, dx, M, max_f_prime, u_fixed, layer, device, is_train=True):
        super(VariantCoeLinear1d, self).__init__()
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
        return torch.pow(u, 2) / (torch.pow(u, 2) + 0.5 * torch.pow(1 - u, 2))

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
        # max_f_prime = round(max_f_prime, 1)
        max_f_prime = round(max_f_prime+0.01, 2)
        return max_f_prime

    def update(self):
        # 计算目前状况下f(u)导数的最大值
        self.u_fixed.requires_grad = True
        max_f_prime = self.df_du(self.u_fixed)
        self.u_fixed.requires_grad = False
        # if 0.001 < max_f_prime  and max_f_prime < 10.0:
        # if 0.04 < max_f_prime  and max_f_prime < 0.3:
        if 0.001 < max_f_prime  and max_f_prime < 3:  # 原来的（0.04，3）
            print("adjust max_f_prime")
            # the old version of numerical scheme
            # # 根据这个最大值， 开始进行delta t 的计算
            # dt_a = self.dx.item()/(max_f_prime + 1e-10)
            # n_time = self.T/dt_a + 100
            # n_time = round(n_time, -2)
            # # print("n_time %.6f" % (n_time))
            # n_time = int(max(500, n_time))
            # dt = self.T/n_time
            # M = max(max_f_prime, 0.05)
            # self.max_f_prime = torch.DoubleTensor(1).fill_(max_f_prime).to(self.device)
            # self.dt = torch.DoubleTensor(1).fill_(dt).to(self.device)
            # self.time_steps = torch.IntTensor(1).fill_(n_time).to(self.device)
            # self.M = torch.DoubleTensor(1).fill_(M).to(self.device)
            # print("\033[32mmax_f_prime %.6f, dt %.6f, time_steps %.6f, m %.6f,\033[0m" % (self.max_f_prime, self.dt, self.time_steps, self.M))
            # if self.dt == 0.08:
            #     self.M = torch.DoubleTensor(1).fill_(0.1).to(self.device)
            # print("\033[32mmax_f_prime %.6f, dt %.6f, time_steps %.6f, m %.6f,\033[0m" % (self.max_f_prime, self.dt, self.time_steps, self.M))

            # the new version of numerical scheme
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



# # test
# def test():
#     batch_size = 1
#     device = 'cpu'
#     T = 40
#     X = 10
#     dt = 0.08
#     dx = 0.05  # 2   # 0.05
#     N = 200  # 5   # 200
#     M = 0.1000
#     time_steps = 200
#     max_f_prime = -0.03
#     u_0_np = np.zeros((1, N), dtype=float)
#     u_0_np[:1, 80:120] = 0.8
#     u_0 = torch.from_numpy(u_0_np)
#     u_0 = u_0.to(device)
#     # 引入 u_fixed, 用来计算max_f_prime
#     du = 1.2/52
#     u_fixed_0 = -0.1+0.5*du
#     u_fixed_np = np.zeros((1, 52), dtype=float)
#     u_fixed_np[:1, 0] = u_fixed_0
#     for i in range(1, 52):
#         u_fixed_np[:1, i] = u_fixed_0 + i * du
#     u_fixed = torch.from_numpy(u_fixed_np)
#     u_fixed = u_fixed.to(device)
#     # model
#     linpdelearner = VariantCoeLinear1d(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
#                                        , dx=dx, M=M, max_f_prime=max_f_prime, u_fixed= u_fixed, layer=20,device=device, is_train=False)
#     # 预测值
#     U = linpdelearner(linpdelearner.u0, 500)
#     np.save('data/u_500_pde' + '.npy', U.detach().to('cpu'))
#
#     # # debug时候需要的计算图和参数的相关信息
#     # MyConvNetVis = make_dot(U, params=dict(linpdelearner.named_parameters()), show_attrs=True, show_saved=True)
#     # MyConvNetVis.format = "png"
#     # MyConvNetVis.directory = "data"
#     # MyConvNetVis.view()
#     # params = list(linpdelearner.coe_params())
#     # k = 0
#     # for i in params:
#     #     l = 1
#     #     print("该层的结构：" + str(list(i.size())))
#     #     for j in i.size():
#     #         l *= j
#     #     print("该层参数和：" + str(l))
#     #     k = k + l
#     # print("总参数数量和：" + str(k))

# generate real data
def generate_real_data(save_file, u0_file, u_fixed_file):
    device = 'cpu'
    T = 40
    X = 10
    dt = 0.08
    dx = 0.025  # 0.0125  # 0.025   # 0.05
    N = 400  # 800  # 400   # 200
    M = 0.1000
    time_steps = 200
    max_f_prime = -0.03
    # u_0
    # batch_size = 1
    # u_0_np = np.zeros((1, N), dtype=float)
    # # u_0_np[:1, 80:120] = 0.8 # N = 200
    # # u_0_np[:1, 320:480] = 0.8  # N = 800
    # # u_0_np[:1, 160:240] = 0.8  # N = 400
    # u_0_np[:1, 160:240] = 1.0  # N = 400
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    # u_0
    batch_size = 2
    u_0_np = np.zeros((batch_size, N), dtype=float)
    u_0_np[:1, 160:240] = 1.0  # N = 400
    u_0_np[1:2, 160:240] = 0.6
    u_0 = torch.from_numpy(u_0_np)
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
    linpdelearner = VariantCoeLinear1d(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
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
    np.save(u0_file, u_0.detach().to('cpu'))
    np.save(u_fixed_file, u_fixed.detach().to('cpu'))



if __name__ == "__main__":
    # experiment_name = 'N_400_example_2_dt_0.6_layer_20_beta_0.5'
    # real_data_file = 'data/' + experiment_name + '_U' + '.npy'
    # u0_file = 'data/' + experiment_name + '_u0' + '.npy'
    # u_fixed_file = 'data/' + experiment_name + '_u_fixed' + '.npy'
    # generate_real_data(real_data_file, u0_file, u_fixed_file)

    # 对数据添加一些噪音处理
    experiment_name = 'N_400_example_2_dt_0.6_layer_20_beta_0.5'
    real_data_file = 'data/' + experiment_name + '_U' + '.npy'

    obs_data = torch.from_numpy(np.load(real_data_file))
    shape = obs_data.shape

    U = np.empty(shape=(shape[0], shape[1], shape[2]))
    # 第一种增加噪音的方式
    # for i in range(0, shape[0]):
    #     for k in range(0, shape[1]):
    #         for j in range(0, shape[2]):
    #             if obs_data[i, k, j] != 0:
    #                 U[i, k, j] = obs_data[i, k, j] * (1 + random.uniform(-3, 3) * 0.01)
    #             else:
    #                 U[i, k, j] = random.uniform(-3, 3) * 0.01
    # 第二种增加噪音的方式
    for i in range(0, shape[0]):
        for k in range(0, shape[1]):
            for j in range(0, shape[2]):
                U[i, k, j] = obs_data[i, k, j] + random.uniform(-3, 3) * 0.01

    U_tensor = torch.from_numpy(U).float()
    save_file = 'data/' + experiment_name + '_noise_0.03' + '_U' + '.npy'
    np.save(save_file, U_tensor.detach().to('cpu'))

    # u0
    u_0_data = U_tensor[0, :, :]
    save_file = 'data/' + experiment_name + '_noise_0.03' + '_u0' + '.npy'
    np.save(save_file, u_0_data.detach().to('cpu'))

    # u_fixed
    u_fixed_file = 'data/' + experiment_name + '_u_fixed' + '.npy'
    u_fixed_data = torch.from_numpy(np.load(u_fixed_file))
    save_file = 'data/' + experiment_name + '_noise_0.03' + '_u_fixed' + '.npy'
    np.save(save_file, u_fixed_data.detach().to('cpu'))




