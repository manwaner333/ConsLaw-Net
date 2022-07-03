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
        # f = (2.614239673407566)*u+(-1.441936273744402)*1+(-1.3034071385655406)*u**4+(1.254811985463532)*u**7+(-0.8590785317730645)*u**3+(0.7809428873919907)*u**6+(-0.6264183030592623)*u**2+(-0.5537281345144629)*u**5+(0.46542578704884435)*u**8+(-0.43880360037305643)*u**10+(-0.3227505331953887)*u**9+(-0.15071379844652885)*u**11+(0.11583602439894662)*u**13+(0.0848061065741193)*u**12+(0.05928335636009831)*u**14+(-0.023527938005036896)*u**17+(-0.02099686090431771)*u**16+(-0.008809374616467906)*u**18+(0.007783430128217845)*u**15+(0.00582288316604682)*u**20+(0.003895300729009736)*u**19+(0.002014984310728911)*u**21+(-0.0008614345096021971)*u**23+(-0.0006998141584819637)*u**22+(-0.00020143791575541332)*u**24+(9.80794293428125e-05)*u**25+(6.808651273844694e-05)*u**26+(-6.568599526023435e-06)*u**28+(5.972725383145595e-06)*u**27+(-2.0492734912846565e-06)*u**29
        # f = (-6.834229332266385)*u**3+(2.646872081621761)*u**4+(2.1957148340849746)*u**2+(2.1770218776442927)*u+(0.7682667392176022)*u**5+(0.4030515706146376)*1+(0.0648329581342991)*u**6+(0.0022488876791136214)*u**7+(2.792806935589932e-05)*u**8
        # f = (21.79634900111228)*u**4+(-21.10548976860509)*u**3+(-10.301214750245114)*u**5+(6.787527847462155)*u**2+(2.4417065138644864)*u**6+(1.6637591333546562)*u+(0.44271297060579684)*1+(-0.2826501690527755)*u**7+(0.01271822023951458)*u**8
        # f = (-4.921324843273608)*u**3+(2.376138260382418)*u+(1.7820253935769925)*u**4+(1.0586948612603395)*u**2+(0.6507773456107627)*u**5+(0.5100653529199363)*1+(0.06531203122984827)*u**6+(0.0026593134080233776)*u**7+(3.850552062478322e-05)*u**8
        # f = (2.992355219210167)*u+(-1.9020609801302528)*u**2+(-1.0432818027918314)*u**3+(0.9300084646760871)*u**4+(0.5132954627216219)*1+(0.030124267816086295)*u**5+(0.0003433005384400041)*u**6+(1.6373365855913076e-06)*u**7
        # f = (-5.003773209557326)*u**3+(2.280676828553506)*u+(1.3342259700172239)*u**2+(1.266458844986153)*u**4+(0.9805842451937651)*u**5+(0.45264255197250686)*1+(0.1583734672062787)*u**6+(0.00998955932629509)*u**7+(0.00022096311630228688)*u**8
        f = (24.629216795317745)*u**4+(-23.425112671798402)*u**3+(-12.110430837865726)*u**5+(7.709845229662649)*u**2+(3.021617990809614)*u**6+(1.5297492322870434)*u+(0.452032985130408)*1+(-0.3701770912083237)*u**7+(0.017666000727127224)*u**8
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
        # 如果　max_f_prime　比　self.max_f_prime大，　需要更新self.dt, self.M, self.max_f_prime 和 self.time_step
        # if self.max_f_prime.item() < max_f_prime and max_f_prime < 1:
        if max_f_prime > 0.2 and max_f_prime < 5:
            print("adjust max_f_prime")
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

# generate real data
def generate_predict_data_cell_1600(save_file):
    device = 'cpu'
    T = 2.0
    X = 10
    dt = 0.023529
    dx = 0.05  # 10/200   # 10/1600  # 2   # 0.05
    N = 1600  # 200  # 200  # 1600   # 200
    M = 0.1000
    time_steps = 200
    max_f_prime = -0.03
    batch_size = 1
    # u_0
    u_0_np = np.zeros((1, N), dtype=float)
    # u_0_np[:1, 0:480] = 1.0
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
    U = linpdelearner(linpdelearner.u0, linpdelearner.time_steps)
    print("U")
    print(U.shape)
    print("u_0")
    print(u_0.shape)
    print("u_fixed")
    print(u_fixed.shape)
    np.save(save_file, U.detach().to('cpu'))


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
    # 用来做训练
    # batch_size = 1
    # # u_0
    # u_0_np = np.zeros((1, N), dtype=float)
    # u_0_np[:1, 0:60] = 1.0
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    # 直接提取
    batch_size = 4
    u_0_file = 'data/' + experiment_name + '_u0' + '.npy'
    u_0 = torch.from_numpy(np.load(u_0_file))
    u_0 = u_0.to(device)
    # 用来做测试
    # batch_size = 3
    # u_0_np = np.zeros((batch_size, N), dtype=float)
    # u_0_np[0:1, 60:120] = 0.8
    # u_0_np[1:2, 80:140] = 0.6
    # u_0_np[2:3, 20:80] = 0.7
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
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
    experiment_name = 'N_200_example_4_dt_0.1_layer_10_beta_120_noise_0.05'
    # experiment_name = 'N_200_example_1_dt_0.1_layer_10_beta_120_extra_test'
    # experiment_name = 'N_200_example_2_dt_0.1_layer_10_beta_120_extra_test'
    # experiment_name = 'N_200_example_2_dt_0.1_layer_10_beta_120_noise_0.05'
    real_data_file = 'data/' + experiment_name + '.npy'
    predict_data_file = 'data/' + 'predict_' + experiment_name + '.npy'
    generate_predict_data_cell_200(predict_data_file)
    # generate_predict_data_cell_1600(predict_data_file)
    # generate_predict_data_cell_1600(real_data_file)




