import numpy as np
import torch
from torch import nn
import pdelearner
from tqdm import tqdm
from torch.autograd import grad
np.random.seed(0)
torch.manual_seed(0)

# 初始化参数
def initexpr(model):
    rhi = model.polys
    for poly in rhi:
        for p in poly.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device)
    return None

def _sparse_loss(model):
    """
    SymNet regularization
    """
    loss = 0
    s = 1e-2
    for p in model.coe_params():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss

def printcoeffs(model):
    for poly in model.polys:
        tsym_0, csym_0, tsym_1, csym_1 = poly.coeffs()
        print(tsym_0[:20])
        print(csym_0[:20])
        print(tsym_1[:20])
        print(csym_1[:20])
        str_molecular = '(' + str(csym_0[0]) + ')' + '*' + str(tsym_0[0])
        for index in range(1, len(tsym_0)):
            str_molecular += '+' + '(' + str(csym_0[index]) + ')' + '*' + str(tsym_0[index])

        str_denominator = '(' + str(csym_1[0]) + ')' + '*' + str(tsym_1[0])
        for index in range(1, len(tsym_1)):
            str_denominator += '+' + '(' + str(csym_1[index]) + ')' + '*' + str(tsym_1[index])

        print(str_molecular)
        print(str_denominator)

# 正式开始训练
def train(namestobeupdate, u_0, u_fixed, obs_data):
    mse = nn.MSELoss()
    # 引入max_f_prime
    max_f_prime = -0.1
    # 引入模型
    linpdelearner = pdelearner.VariantCoeLinear1d(T=namestobeupdate['T'], N=namestobeupdate['N'], X=namestobeupdate['X'],
                                                  batch_size=namestobeupdate['batch_size'], u0=u_0, dt=namestobeupdate['dt'], time_steps=namestobeupdate['time_steps'],
                                                  dx=namestobeupdate['dx'], M=namestobeupdate['M'], max_f_prime=max_f_prime, u_fixed=u_fixed, layer=namestobeupdate['layer'],
                                                  device=namestobeupdate['device'], is_train=True)
    if namestobeupdate['precision'] == 'double':
        linpdelearner.double()
    else:
        linpdelearner.float()
    linpdelearner.to(namestobeupdate['device'])
    # 初始化参数
    initexpr(linpdelearner)
    # 开始训练
    n_epochs = namestobeupdate['n_epochs']
    stepnum = namestobeupdate['layer']
    lr = namestobeupdate['lr']
    opt_model = torch.optim.SGD(linpdelearner.coe_params(), lr=lr)
    # 真实值
    u_obs = obs_data
    for epoch in tqdm(range(n_epochs)):
        # 模型更新
        linpdelearner.update()
        ut = linpdelearner.u0
        stableloss = 0
        dataloss = 0
        sparseloss = _sparse_loss(linpdelearner)

        # 时间list
        obs_t = []
        for i in range(stepnum):
            obs_t.append(0.6 * i)
        # 真实数据， 步骤list
        dt_fixed = 0.169492
        obs_time_step = []

        for ele in obs_t:
            obs_time_step.append(round(ele/dt_fixed))
        obs_data_choose = obs_data[obs_time_step, :, :]

        # 预测数据， 步骤list
        dt_changed = linpdelearner.dt.item()
        pre_time_step = []
        for ele in obs_t:
            pre_time_step.append(round(ele/dt_changed))
        # 预测的轨迹
        trajectories = linpdelearner(ut, pre_time_step[-1] + 1)
        pre_data_choose = trajectories[pre_time_step, :, :]

        linpdelearner.u_fixed.requires_grad = True
        f_test = 0.05 * linpdelearner.f_predict(linpdelearner.u_fixed)
        dfdu = grad(f_test, linpdelearner.u_fixed, grad_outputs=torch.ones_like(f_test), create_graph=False)[0]
        linpdelearner.u_fixed.requires_grad = False
        max_f_prime = torch.max(torch.abs(dfdu))
        stableloss = abs(max_f_prime - 0.1)

        # 打印相关的数据
        print('obs_time_step:')
        print(obs_time_step)
        # print('obs_data_choose:')
        # print(obs_data_choose)
        print("dt")
        print(dt_changed)
        print('pre_time_step:')
        print(pre_time_step)
        # print('pre_data_choose:')
        # print(pre_data_choose)

        # for step in range(1, stepnum, 1):
        #     dataloss
        #     # dataloss = dataloss + torch.mean(torch.abs(obs_data_choose[step]-pre_data_choose[step]))
        dataloss = mse(obs_data_choose[:, 0, :], pre_data_choose[:, 0, :])

        print("\033[33mdata loss0 %.6f, stable loss %.6f, sparse loss %.6f, max_f_prime loss %.6f, \033[0m" % (dataloss, 0.05*stableloss, 0.005*sparseloss, max_f_prime, )) # 黄色
        error_batch = dataloss + 0.0 * sparseloss + 0.0 * stableloss
        error_batch.backward()
        opt_model.step()
        if max_f_prime > 0.09 and max_f_prime <= 0.1:
            printcoeffs(linpdelearner)

    # 打印出相关参数
    printcoeffs(linpdelearner)




def main():

    options = {
        '--device': 'cpu',   # 'cuda:0',
        '--precision': 'double',
        '--taskdescriptor': 'N_200_example_7_dt_0.1_layer_10_beta_negative_100_adjust_du',
        '--batch_size': 7,  # 250,  # 150,
        '--maxiter': 255,   # 156,
        '--X': 10,
        '--T': 2.0,
        '--dx': 10/200,  # 0.0125,   # 0.025,  # 0.125,
        '--N': 200,    # 800,  # 400,    # cell number
        '--M': 0.1100,
        '--dt': 0.169492,
        '--time_steps': 500,
        '--layer': 20,  # time steps
        '--n_epochs': 500,
        '--lr': 0.005,
        '--max_order': 4,
        '--xn': '50',
        '--yn': '50',
        '--interp_degree': 4,
        '--interp_mesh_size': 20,
        '--initfreq': 4,
        '--constraint': 'moment',
        '--recordfile': 'convergence',
        '--recordcycle': 10,
        '--savecycle': 10000,
        '--repeatnum': 25,
        '--teststepnum': 80,
        '--variant_coe_magnitude': 1,
        '--nonlinear_interp_degree': 4,
        '--nonlinear_interp_mesh_size': [20],
        '--nonlinear_interp_mesh_bound': [[0], [0.999]],
        '--start_noise_level': 0.015,
        '--end_noise_level': 0.015,
    }
    namestobeupdate = {}
    namestobeupdate['device'] = options['--device']
    namestobeupdate['precision'] = options['--precision']
    namestobeupdate['taskdescriptor'] = options['--taskdescriptor']
    namestobeupdate['max_order'] = options['--max_order']
    namestobeupdate['xn'] = options['--xn']
    namestobeupdate['yn'] = options['--yn']
    namestobeupdate['interp_degree'] = options['--interp_degree']
    namestobeupdate['interp_mesh_size'] = [options['--interp_mesh_size']]
    namestobeupdate['initfreq'] = options['--initfreq']
    namestobeupdate['batch_size'] = options['--batch_size']
    namestobeupdate['maxiter'] = options['--maxiter']
    namestobeupdate['T'] = options['--T']
    namestobeupdate['X'] = options['--X']
    namestobeupdate['dt'] = options['--dt']
    namestobeupdate['dx'] = options['--dx']
    namestobeupdate['time_steps'] = options['--time_steps']
    namestobeupdate['layer'] = options['--layer']
    namestobeupdate['n_epochs'] = options['--n_epochs']
    namestobeupdate['lr'] = options['--lr']
    namestobeupdate['N'] = options['--N']
    namestobeupdate['M'] = options['--M']
    namestobeupdate['layer'] = options['--layer']
    namestobeupdate['constraint'] = options['--constraint']
    namestobeupdate['recordfile'] = options['--recordfile']
    namestobeupdate['recordcycle'] = options['--recordcycle']
    namestobeupdate['savecycle'] = options['--savecycle']
    namestobeupdate['repeatnum'] = options['--repeatnum']
    namestobeupdate['teststepnum'] = options['--teststepnum']
    namestobeupdate['variant_coe_magnitude'] = options['--variant_coe_magnitude']
    namestobeupdate['nonlinear_interp_degree'] = options['--nonlinear_interp_degree']
    namestobeupdate['nonlinear_interp_mesh_size'] = options['--nonlinear_interp_mesh_size']
    namestobeupdate['nonlinear_interp_mesh_bound'] = options['--nonlinear_interp_mesh_bound']
    namestobeupdate['start_noise_level'] = options['--start_noise_level']
    namestobeupdate['end_noise_level'] = options['--end_noise_level']

    # 引入数据
    # 引入u_0
    u_0_file = 'data/' + namestobeupdate['taskdescriptor'] + '_u0' + '.npy'
    u_0 = torch.from_numpy(np.load(u_0_file))
    u_0 = u_0.to(namestobeupdate['device'])

    # 引入 u_fixed, 用来计算max_f_prime
    u_fixed_file = 'data/' + namestobeupdate['taskdescriptor'] + '_u_fixed' + '.npy'
    u_fixed = torch.from_numpy(np.load(u_fixed_file))
    u_fixed = u_fixed.to(namestobeupdate['device'])

    # load real data
    real_data_file = 'data/' + namestobeupdate['taskdescriptor'] + '_U' + '.npy'
    obs_data = torch.from_numpy(np.load(real_data_file))
    obs_data = obs_data.to(namestobeupdate['device'])
    # train
    train(namestobeupdate, u_0, u_fixed, obs_data)



if __name__ == "__main__":
    main()