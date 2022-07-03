#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import numpy as np
import scipy
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize import fmin_bfgs as bfgs
import torch
import torchvision
import aTEAM
from aTEAM.optim import NumpyFunctionInterface
import pdelearner
import linpdeconfig
np.random.seed(0)
torch.manual_seed(0)
from scipy.optimize import check_grad
from torch.autograd import Variable
# from torchviz import make_dot


options = {
    '--device': 'cpu',  #'cpu',    # 'cuda:0',
    '--precision': 'double',
    '--taskdescriptor': 'N_200_example_1_dt_0.1_layer_10_beta_10',
    '--batch_size': 1,  # 250,  # 150,
    '--maxiter': 255,   # 156,
    '--X': 10,
    '--T': 2.0,
    '--dx': 0.05,  # 0.0125,   # 0.025,  # 0.125,
    '--N': 200,    # 800,  # 400,    # cell number
    '--M': 1.58,
    '--dt': 0.023529,
    '--time_steps': 900,
    '--layer': 10,  # time steps
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

options = linpdeconfig.setoptions(argv=sys.argv[1:], kw=options, configfile=None)
namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)
globals().update(namestobeupdate)

# init parameters
def initexpr(model):
    rhi = model.polys
    for poly in rhi:
        for p in poly.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 3
                     # * 1e-1 * 6
    return None
initexpr(linpdelearner)

# def initexpr(model):
#     rhi = model.polys
#     rhi[0].layer0.weight.data[0, 0] = -0.0131
#     rhi[0].layer0.weight.data[1, 0] = 0.0533
#     rhi[0].layer0.bias.data[0] = 0.0208
#     rhi[0].layer0.bias.data[1] = 0.0420
#
#     rhi[0].layer1.weight.data[0, 0] = -0.0413
#     rhi[0].layer1.weight.data[1, 0] = -0.0476
#     rhi[0].layer1.weight.data[0, 1] = -0.0397
#     rhi[0].layer1.weight.data[1, 1] = 0.0186
#     rhi[0].layer1.bias.data[0] = 0.0204
#     rhi[0].layer1.bias.data[1] = 0.0711
#
#     rhi[0].molecular.weight.data[0, 0] = 0.0073
#     rhi[0].molecular.weight.data[0, 1] = -0.0335
#     rhi[0].molecular.weight.data[0, 2] = -0.0107
#     rhi[0].molecular.bias.data[0] = -0.0213
#
#     rhi[0].denominator.weight.data[0, 0] = -0.0352
#     rhi[0].denominator.weight.data[0, 1] = -0.0053
#     rhi[0].denominator.weight.data[0, 2] = -0.0621
#     rhi[0].denominator.bias.data[0] = -0.0245
#     return None



for name, parameters in linpdelearner.named_parameters():
    print(name, ':', parameters)


params = list(linpdelearner.coe_params())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))


for l in [layer]:
    if l == 0:
        callback.stage = 'warmup'
        isfrozen = (False if constraint == 'free' else True)
    else:
        callback.stage = 'layer-'+str(l)
        if constraint == 'moment' or constraint == 'free':
            isfrozen = False
        elif constraint == 'frozen':
            isfrozen = True
    stepnum = (l if l>=1 else 1)
    # load real data
    real_data_file = 'data/' + taskdescriptor + '_U' + '.npy'
    obs_data = torch.from_numpy(np.load(real_data_file))
    obs_data = obs_data.to(device)

    def forward():
        sparsity = 0.00
        stablize = 0.00
        dataloss, sparseloss, stableloss = linpdeconfig.loss(linpdelearner, stepnum, obs_data)
        loss = dataloss + sparsity * sparseloss + stablize * stableloss
        # loss = stableloss
        return loss

    nfi = NumpyFunctionInterface(
        [dict(params=linpdelearner.coe_params(), isfrozen=False, x_proj=None, grad_proj=None)],
        forward=forward, always_refresh=False)
    callback.nfi = nfi


    xopt, f, d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, callback=callback, factr=1e0, pgtol=1e-16, maxiter=maxiter, iprint=1,)
    nfi.flat_param = xopt
    callback.save(xopt, 'final')

    for name, parameters in linpdelearner.named_parameters():
        print(name, ':', parameters)

    def printcoeffs():
        with callback.open() as output:
            print('current expression:', file=output)
            for poly in linpdelearner.polys:
                tsym_0, csym_0, tsym_1, csym_1 = poly.coeffs()
                print(tsym_0[:20], file=output)
                print(csym_0[:20], file=output)
                print(tsym_1[:20], file=output)
                print(csym_1[:20], file=output)
                str_molecular = '(' + str(csym_0[0]) + ')' + '*' + str(tsym_0[0])
                for index in range(1, len(tsym_0)):
                    str_molecular += '+' + '(' + str(csym_0[index]) + ')' + '*' + str(tsym_0[index])

                str_denominator = '(' + str(csym_1[0]) + ')' + '*' + str(tsym_1[0])
                for index in range(1, len(tsym_1)):
                    str_denominator += '+' + '(' + str(csym_1[index]) + ')' + '*' + str(tsym_1[index])

                print(str_molecular)
                print(str_denominator)

    printcoeffs()
    print(d['warnflag'])
    print(d['task'])

    # try:
    #     # optimize
    #     xopt,f,d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, callback=callback, factr=1e0,pgtol=1e-16,maxiter=maxiter,iprint=50)
    # except RuntimeError as Argument:
    #     with callback.open() as output:
    #         print(Argument, file=output) # if overflow then just print and continue
    # finally:
    #     # save parameters
    #     nfi.flat_param = xopt
    #     callback.save(xopt, 'final')


