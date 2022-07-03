#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
import torch
from aTEAM.optim import NumpyFunctionInterface
import linpdeconfig
np.random.seed(0)
torch.manual_seed(0)
# from torchviz import make_dot

options = {
    '--device': 'cpu',  #'cpu',    # 'cuda:0',
    '--precision': 'double',
    '--taskdescriptor': 'N_400_example_2_dt_0.1_layer_10_beta_0.5',
    '--batch_size': 2,  # 250,  # 150,
    '--maxiter': 800,   # 156,
    '--X': 10,
    '--T': 2.0,
    '--dx': 0.025,   # 10/1600,    # 0.05,
    '--N': 400,   # 1600,     # 200,    # cell number
    '--M': 0.1100,
    '--dt': 0.169492,
    '--time_steps': 236,
    '--layer': 10,  # time steps
    '--recordfile': 'convergence',
    '--recordcycle': 10,
    '--savecycle': 10000,
}

options = linpdeconfig.setoptions(argv=sys.argv[1:], kw=options, configfile=None)
namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)
globals().update(namestobeupdate)

# init parameters
def initexpr(model):
    rhi = model.polys
    for poly in rhi:
        for p in poly.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 6
    return None
initexpr(linpdelearner)

# 参数情况统计
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
    callback.stage = 'layer-'+str(l)
    isfrozen = False
    stepnum = (l if l>=1 else 1)
    # load real data
    real_data_file = 'data/' + taskdescriptor + '_U' + '.npy'
    obs_data = torch.from_numpy(np.load(real_data_file))
    obs_data = obs_data.to(device)

    def forward():
        sparsity = 0.0
        # stablize = 0.005
        stablize = 0.0
        dataloss, sparseloss, stableloss = linpdeconfig.loss(linpdelearner, stepnum, obs_data)
        loss = dataloss + sparsity * sparseloss + stablize * stableloss
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
                tsym,csym = poly.coeffs()
                print(tsym[:20], file=output)
                print(csym[:20], file=output)
                str_molecular = '(' + str(csym[0]) + ')' + '*' + str(tsym[0])
                for index in range(1, len(tsym)):
                    str_molecular += '+' + '(' + str(csym[index]) + ')' + '*' + str(tsym[index])
                print(str_molecular)

    printcoeffs()
    print(d['warnflag'])
    print(d['task'])

