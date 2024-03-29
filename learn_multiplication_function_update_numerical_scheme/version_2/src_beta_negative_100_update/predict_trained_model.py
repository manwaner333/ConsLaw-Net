import numpy as np
import torch
import linpdeconfig

# 加载模型
experiment_name = 'dx_0.05_example_1_layer_30_jump_1_lambda_10_test'
configfile = 'checkpoint/' + experiment_name + '/options.yaml'
train_layers = 30
options = linpdeconfig.setoptions(configfile=configfile, isload=True)
namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)
globals().update(namestobeupdate)
callback.load(train_layers)
model = linpdelearner

# 给模型重新赋一些变量的值: u0,
model.N = 1600
model.batch_size = 1
dx = 10/1600
model.dx = torch.DoubleTensor(1).fill_(dx)
u_0_np = np.zeros((1, model.N), dtype=float)
# u_0_np[:1, 0:40] = 1.0
u_0_np[:1, 0:480] = 1.0
u_0 = torch.from_numpy(u_0_np)
u_0 = u_0.to(namestobeupdate['device'])
model.u0 = u_0
max_f_prime = torch.DoubleTensor(1).fill_(-0.1)
model.max_f_prime = max_f_prime

# params = list(model.coe_params())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i)))

# 预测
test_time_steps = 500
trajectories = model(u_0, test_time_steps)
print("new time_steps")
print(model.time_steps)
print("max value of prediction data:")
print(torch.max(trajectories[499, :, :]))

# 保存预测数据
test_data_name = experiment_name
np.save('data/' + 'predict_' + test_data_name + 'new_initial' + '.npy', trajectories.data)





