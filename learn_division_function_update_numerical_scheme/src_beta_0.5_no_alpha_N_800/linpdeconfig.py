#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, contextlib
import numpy as np
import torch
from torch.autograd import Variable
import getopt, yaml, time
import pdelearner
from torch.autograd import grad
np.random.seed(0)
torch.manual_seed(0)
from torch import nn
mse = nn.MSELoss()

def _options_cast(options, typeset, thistype):
    for x in typeset:
        options['--'+x] = thistype(options['--'+x])
    return options
def _option_analytic(option, thistype):
    if not isinstance(option, str):
        return option
    l0 = option.split(',')
    l = []
    for l1 in l0:
        try:
            ll = thistype(l1)
            x = [ll,]
        except ValueError:
            z = l1.split('-')
            x = list(range(int(z[0]), int(z[1])+1))
        finally:
            l = l+x
    return l
def _setoptions(options):
    assert options['--precision'] in ['float','double']
    # str options
    strtype = ['taskdescriptor',  'constraint', 'recordfile', 'device']
    options = _options_cast(options, strtype, str)
    assert options['--constraint'] in ['frozen', 'moment', 'free']
    # int options
    inttype = ['max_order', 'xn', 'yn', 'interp_degree', 'interp_mesh_size',
               'initfreq', 'batch_size', 'teststepnum', 'maxiter', 'recordcycle', 'savecycle', 'repeatnum', 'time_steps', 'layer']
    options = _options_cast(options, inttype, int)
    # float options
    floattype = ['dt', 'variant_coe_magnitude', 'start_noise_level', 'end_noise_level', 'T', 'X']
    options = _options_cast(options, floattype, float)
    # options['--layer'] = list(_option_analytic(options['--layer'], int))
    return options

def setoptions(*, argv=None, kw=None, configfile=None, isload=False):
    """
    proirity: argv>kw>configfile
    Arguments:
        argv (list): command line options
        kw (dict): options
        configfile (str): configfile path
        isload (bool): load or set new options
    """
    options = {
        '--precision': 'double',
        '--xn': '50',
        '--yn': '50',
    }
    longopts = list(k[2:]+'=' for k in options)
    longopts.append('configfile=')
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f', longopts=longopts)[0]))
    if '--configfile' in options:
        assert configfile is None, 'duplicate configfile in argv.'
        configfile = options['--configfile']
    if not configfile is None:
        options['--configfile'] = configfile
        with open(configfile, 'r') as f:
            options.update(yaml.safe_load(f))
    if not kw is None:
        options.update(kw)
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f', longopts=longopts)[0]))
    options = _setoptions(options)
    options.pop('-f', 1)
    savepath = 'checkpoint/'+options['--taskdescriptor']
    if not isload:
        try:
            os.makedirs(savepath)
        except FileExistsError:
            os.rename(savepath, savepath+'-'+str(np.random.randint(2**32)))
            os.makedirs(savepath)
        with open(savepath+'/options.yaml', 'w') as f:
            print(yaml.dump(options), file=f)
    return options

class callbackgen(object):
    def __init__(self, options, nfi=None, module=None, stage=None):
        self.taskdescriptor = options['--taskdescriptor']
        self.recordfile = options['--recordfile']
        self.recordcycle = options['--recordcycle']
        self.savecycle = options['--savecycle']
        self.savepath = 'checkpoint/'+self.taskdescriptor
        self.startt = time.time()
        self.Fs = []
        self.Gs = []
        self.ITERNUM = 0

    @property
    def stage(self):
        return self._stage
    @stage.setter
    def stage(self, v):
        self._stage = v
        with self.open() as output:
            print('\n', file=output)
            print('current stage is: '+v, file=output)
    @contextlib.contextmanager
    def open(self):
        isfile = (not self.recordfile is None)
        if isfile:
            output = open(self.savepath+'/'+self.recordfile, 'a')
        else:
            output = sys.stdout
        try:
            yield output
        finally:
            if isfile:
                output.close()

    # remember to set self.nfi,self.module,self.stage
    def save(self, xopt, iternum):
        self.nfi.flat_params = xopt
        try:
            os.mkdir(self.savepath+'/params')
        except:
            pass
        filename = self.savepath+'/params/'+str(self.stage)+'-xopt-'+str(iternum)
        torch.save(self.module.state_dict(), filename)
        return None
    def load(self, l, iternum=None):
        """
        load storaged parameters from a file.
        the name of the file from which we will load
        is determined by l and iternum
        """
        if l == 0:
            stage = 'warmup'
        else:
            stage = 'layer-'+str(l)
        if iternum is None:
            iternum = 'final'
        else:
            iternum = str(iternum)
        filename = self.savepath+'/params/'+str(stage)+'-xopt-'+iternum
        # params = torch.load(filename, map_location=self.module.device)
        params = torch.load(filename)
        self.module.load_state_dict(params)
        return None

    def record(self, xopt, iternum, **args):
        self.Fs.append(self.nfi.f(xopt))
        self.Gs.append(np.linalg.norm(self.nfi.fprime(xopt)))
        stopt = time.time()
        with self.open() as output:
            print('iter:{:6d}'.format(iternum), '   time: {:.2f}'.format(stopt-self.startt), file=output)
            print('Func: {:.2e}'.format(self.Fs[-1]), ' |g|: {:.2e}'.format(self.Gs[-1]), file=output)
        self.startt = stopt
        return None
    def __call__(self, xopt, **args):
        if self.ITERNUM%self.recordcycle == 0:
            self.record(xopt, iternum=self.ITERNUM, **args)
        if self.ITERNUM%self.savecycle == 0:
            self.save(xopt, iternum=self.ITERNUM)
        self.ITERNUM += 1
        return None

def setenv(options):

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

    # 引入u_0
    u_0_file = 'data/' + namestobeupdate['taskdescriptor'] + '_u0' + '.npy'
    u_0 = torch.from_numpy(np.load(u_0_file))
    u_0 = u_0.to(namestobeupdate['device'])

    # 引入 u_fixed, 用来计算max_f_prime
    u_fixed_file = 'data/' + namestobeupdate['taskdescriptor'] + '_u_fixed' + '.npy'
    u_fixed = torch.from_numpy(np.load(u_fixed_file))
    u_fixed = u_fixed.to(namestobeupdate['device'])
    # 引入max_f_prime
    max_f_prime = -0.1
    linpdelearner = pdelearner.VariantCoeLinear1d(T=namestobeupdate['T'], N=namestobeupdate['N'], X=namestobeupdate['X'],
                                                  batch_size=namestobeupdate['batch_size'], u0=u_0, dt=namestobeupdate['dt'], time_steps=namestobeupdate['time_steps'],
                                                  dx=namestobeupdate['dx'], M=namestobeupdate['M'], max_f_prime=max_f_prime,  u_fixed=u_fixed, layer=namestobeupdate['layer'],
                                                  device=namestobeupdate['device'], is_train=True)

    if namestobeupdate['precision'] == 'double':
        linpdelearner.double()
    else:
        linpdelearner.float()

    linpdelearner.to(namestobeupdate['device'])
    callback = callbackgen(options)  # some useful interface
    callback.module = linpdelearner

    return namestobeupdate, callback, linpdelearner

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


# # 如果观测值是连续的
# def loss(model, stepnum, layerweight=None):
#     # 注意这个地方 stepnum 和 model.time_steps 是不一样的
#     if layerweight is None:
#         layerweight = [1,]*stepnum
#         layerweight[-1] = 1
#     ut = model.u0
#     stableloss = 0
#     sparseloss = _sparse_loss(model)
#     trajectories = model(ut, stepnum)
#     # 模型调用完之后，如果model.U 需要调整， 现在状态下已经调整好了
#     u_obs = model.U
#     # print(u_obs[19, 0, 80:120])
#     # dataloss_1 = torch.mean(torch.abs(trajectories[1:, :, :]-u_obs[1:stepnum, :, :]))
#     dataloss = 0
#     for step in range(1, stepnum):
#         uttmp = trajectories[step, :, :]
#         dataloss = dataloss + torch.mean(torch.abs(uttmp-u_obs[step]))
#     print("data loss0 %.6f," % (dataloss))
#     return dataloss, sparseloss, stableloss



# # # 固定时间点, 但是利用四舍五入的方法进行近似, 由于m是不断变化的， 因此观测值也是需要调整的
# def loss(model, stepnum, obs_data, layerweight=None):
#     # 注意这个地方 stepnum 和 model.time_steps 是不一样的
#     if layerweight is None:
#         layerweight = [1,]*stepnum
#         layerweight[-1] = 1
#     ut = model.u0
#     stableloss = 0
#     dataloss = 0
#     sparseloss = _sparse_loss(model)
#     # 真实值
#     u_obs = obs_data
#     # 模型更新
#     model.update()
#     # 求出模型的dt和真实的dt之间的倍数关系
#     dt_fixed = 0.169492
#     dt_changed = model.dt.item()
#     factor = dt_fixed/dt_changed
#
#     # 需要预测的时间长度
#     stepnum_adjust = round(factor * stepnum)
#     # 预测的轨迹
#     trajectories = model(ut, stepnum_adjust + 1)
#
#     model.u_fixed.requires_grad = True
#     f_test = 0.05 * model.f_predict(model.u_fixed)
#     dfdu = grad(f_test, model.u_fixed, grad_outputs=torch.ones_like(f_test), create_graph=False)[0]
#     model.u_fixed.requires_grad = False
#     max_f_prime = torch.max(torch.abs(dfdu))
#
#     # max_f_prime = model.max_f_prime
#     stableloss = abs(max_f_prime - 0.1)
#
#     for step in range(1, stepnum, 1):
#         step_adjust = factor * step
#         step_adjust = round(step_adjust)
#         if (step == 1):
#             print("\033[32mstepnum %.6f, step %.6f, step_adjust %.6f, \033[0m" % (stepnum, step, step_adjust)) # 绿色
#         uttmp = trajectories[step_adjust, :, :]
#         dataloss = dataloss + torch.mean(torch.abs(uttmp-u_obs[step]))
#     print("\033[33mdata loss0 %.6f, stable loss %.6f, sparse loss %.6f, max_f_prime loss %.6f, factor %.6f, \033[0m" % (dataloss, 0.05*stableloss, 0.005*sparseloss, max_f_prime, factor)) # 黄色
#     return dataloss, sparseloss, stableloss


# 修改成按固定时间点的loss
def loss(model, stepnum, obs_data, layerweight=None):
    # 注意这个地方 stepnum 和 model.time_steps 是不一样的
    if layerweight is None:
        layerweight = [1,]*stepnum
        layerweight[-1] = 1
    ut = model.u0
    stableloss = 0
    dataloss = 0
    sparseloss = 0
    # sparseloss = _sparse_loss(model)
    # 模型更新
    model.update()
    # 时间list
    obs_t = []
    for i in range(stepnum):
        obs_t.append(0.1 * i)
    # 真实数据， 步骤list
    dt_fixed = 0.004474
    obs_time_step = []

    for ele in obs_t:
        obs_time_step.append(round(ele/dt_fixed))
    obs_data_choose = obs_data[obs_time_step, :, :]

    # 预测数据， 步骤list
    dt_changed = model.dt.item()
    pre_time_step = []
    for ele in obs_t:
        pre_time_step.append(round(ele/dt_changed))
    # 预测的轨迹
    trajectories = model(ut, pre_time_step[-1] + 1)
    pre_data_choose = trajectories[pre_time_step, :, :]

    model.u_fixed.requires_grad = True
    f_test = model.f_predict(model.u_fixed)
    dfdu = grad(f_test, model.u_fixed, grad_outputs=torch.ones_like(f_test), create_graph=False)[0]
    model.u_fixed.requires_grad = False
    max_f_prime = torch.max(torch.abs(dfdu))
    # stableloss = abs(max_f_prime - 0.1)

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
    #     dataloss = dataloss + torch.mean(torch.abs(obs_data_choose[step]-pre_data_choose[step]))
    dataloss = mse(obs_data_choose[:, :, :], pre_data_choose[:, :, :])
    print("\033[33mdata loss0 %.6f, stable loss %.6f, sparse loss %.6f, max_f_prime loss %.6f, \033[0m" % (dataloss, 0.05*stableloss, 0.005*sparseloss, max_f_prime, )) # 黄色
    return dataloss, sparseloss, stableloss




# # # # 固定时间点, 但是利用四舍五入的方法进行近似, 由于m是不断变化的， 因此观测值也是需要调整的
# def loss(model, stepnum, obs_data, layerweight=None):
#     # 注意这个地方 stepnum 和 model.time_steps 是不一样的
#     if layerweight is None:
#         layerweight = [1,]*stepnum
#         layerweight[-1] = 1
#     ut = model.u0
#     stableloss = 0
#     dataloss = 0
#     sparseloss = 0
#
#     model.u_fixed.requires_grad = True
#     f_test = 0.05 * model.f_predict(model.u_fixed)
#     dfdu = grad(f_test, model.u_fixed, grad_outputs=torch.ones_like(f_test), create_graph=True)[0]
#     model.u_fixed.requires_grad = False
#     max_f_prime = torch.max(torch.abs(dfdu))
#     print("max_f_prime")
#     print(max_f_prime)
#
#     stableloss = abs(max_f_prime - 0.1)
#     if stableloss >= 0.04:
#         print("\033[31mdata loss0 %.6f, sparse loss %.6f, stable loss %.6f, \033[0m" % (dataloss, sparseloss, stableloss))
#         return dataloss, sparseloss, stableloss * 200
#     else:
#         sparseloss = _sparse_loss(model)
#         # 真实值
#         u_obs = obs_data
#         # 求出模型的dt和真实的dt之间的倍数关系
#         dt_fixed = 0.169492
#         dt_changed = model.dt.item()
#         factor = dt_fixed/dt_changed
#         print('factor....')
#         print(factor)
#         if (max_f_prime >= 0.07 and max_f_prime <= 0.12):
#             model.dt = torch.DoubleTensor(1).fill_(dt_fixed).to(model.device)
#             factor = 1
#         # 需要预测的时间长度
#         stepnum_adjust = round(factor * stepnum)
#         # 预测的轨迹
#         trajectories = model(ut, stepnum_adjust + 1)
#         for step in range(1, stepnum, 1):
#             step_adjust = factor * step
#             step_adjust = round(step_adjust)
#             # if (step == 1):
#             print("\033[32mstepnum %.6f, step %.6f, step_adjust %.6f, \033[0m" % (stepnum, step, step_adjust))
#             uttmp = trajectories[step_adjust, :, :]
#             dataloss = dataloss + torch.mean(torch.abs(uttmp-u_obs[step]))
#         print("\033[33mdata loss0 %.6f, sparse loss %.6f, stable loss %.6f, \033[0m" % (dataloss,  0.005 * sparseloss, stableloss))
#         return dataloss, sparseloss, stableloss



