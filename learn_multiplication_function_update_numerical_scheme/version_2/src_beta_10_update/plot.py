import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torchvision
import linpdeconfig
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pltutils import *
import seaborn as sns

# sns.set_theme()

# def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
#     fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
#     g1 = sns.heatmap(real_data[:, example_index, :].T, cmap="YlGnBu", cbar=True, ax=axes[0])
#     g1.set_ylabel('cell')
#     g1.set_xlabel('time_step')
#     g2 = sns.heatmap(predict_data[:, example_index, :].T, cmap="YlGnBu", cbar=True, ax=axes[1])
#     g2.set_ylabel('cell')
#     g2.set_xlabel('time_step')
#     g3 = sns.heatmap(real_data[:, example_index, :].T - predict_data[:, example_index, :].T
#                      , cmap="YlGnBu", cbar=True, ax=axes[2])
#     g3.set_ylabel('cell')
#     g3.set_xlabel('time_step')
#     plt.show()
#
#     save_dir = 'figures/' + name
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     file_name = '/real_predict_loss_example' + '_' + str(example_index) + '_' + '.pdf'
#     fig.savefig(save_dir + file_name)


def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
    fig, axes = plt.subplots(1, 2, figsize=(50, 16), sharey=True)
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50
    cbar_size = 50
    g1 = sns.heatmap(np.flip(real_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, annot_kws={"size":30}, ax=axes[0])
    g1.set_ylabel('T', fontsize=label_fontsize)
    g1.set_xlabel('X', fontsize=label_fontsize)
    g1.set_xticklabels([])
    g1.set_yticklabels([])
    # plt.xticks(np.arange(0, 2, step=0.2),list('abcdefghigk'),rotation=45)
    g1.set_title("exact u(x, t)", fontsize=title_fontsize)
    cax1 = plt.gcf().axes[-1]
    cax1.tick_params(labelsize=cbar_size)
    cax1.spines['top'].set_linewidth(linewith_frame)
    cax1.spines['bottom'].set_linewidth(linewith_frame)
    cax1.spines['right'].set_linewidth(linewith_frame)
    cax1.spines['left'].set_linewidth(linewith_frame)

    g2 = sns.heatmap(np.flip(predict_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, ax=axes[1])
    g2.set_ylabel('T', fontsize=label_fontsize)
    g2.set_xlabel('X', fontsize=label_fontsize)
    g2.set_xticklabels([])
    g2.set_yticklabels([])
    g2.set_title("prediction u(x, t)", fontsize=title_fontsize)
    cax2 = plt.gcf().axes[-1]
    cax2.tick_params(labelsize=cbar_size)
    cax2.spines['top'].set_linewidth(linewith_frame)
    cax2.spines['bottom'].set_linewidth(linewith_frame)
    cax2.spines['right'].set_linewidth(linewith_frame)
    cax2.spines['left'].set_linewidth(linewith_frame)



    # g3 = sns.heatmap(np.flip(real_data[:, example_index, :], 0) - np.flip(predict_data[:, example_index, :], 0)
    #                  , cmap="YlGnBu", cbar=True, ax=axes[2])
    # g3.set_ylabel('T', fontsize=label_fontsize)
    # g3.set_xlabel('X', fontsize=label_fontsize)
    # g3.set_xticklabels([])
    # g3.set_yticklabels([])
    # g3.set_title("error u(x, t)", fontsize=title_fontsize)
    # cax = plt.gcf().axes[-1]
    # cax.tick_params(labelsize=cbar_size)

    # plt.show()

    save_dir = 'figures/' + name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/real_predict_data_example' + '_' + str(example_index) + '_all_time' + '.pdf'
    fig.savefig(save_dir + file_name)


def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index,cell_numbers, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=['exact', 'prediction'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' +'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_predict_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, time_steps_real, time_steps_predict, time, example_index,cell_numbers, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_real_no_noise_data = real_no_noise_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_no_noise_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
    plot_3 = plt.scatter(x_label, fix_timestep_real_data, c="black")
    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim(-0.06, 1.06)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2, plot_3],
               labels=['exact', 'prediction', 'noise'],
               loc="upper right", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' +'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_predict_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def f_real_function(u):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * 10 * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f
def f_predict_function(u):
    # f = (1.572703035473332)*u+(-0.8901902497722335)*u**3+(0.4633220076325746)*1+(0.2512312565455811)*u**4+(0.07791027848965411)*u**2+(-0.00855631627669679)*u**5+(0.0001042457973669932)*u**6
    # f = (1.566360417765867)*u+(-1.0390526598823036)*u**3+(0.5298342125846519)*1+(0.334013481204859)*u**4+(0.16250512597581745)*u**2+(-0.011160669756990205)*u**5+(0.00013468032357803)*u**6
    # f = (1.5768902674323668)*u+(-1.3956738070143373)*u**3+(0.677054194839409)*u**4+(0.49987622212253774)*1+(0.26935583837039323)*u**2+(-0.117246545347296)*u**5+(0.008909303952340627)*u**6+(-0.0002970497883972936)*u**7+(3.5668991236778326e-06)*u**8
    # f = (1.7128265165309222)*u+(-0.5509923917297209)*u**2+(0.4872594462534223)*1+(-0.2165769763150269)*u**3+(0.09348185257210545)*u**4+(-0.004373246557456879)*u**5+(7.160752249045991e-05)*u**6
    # f = (1.8234232183483077)*u+(-0.8203779761939968)*u**2+(0.5501892425816355)*1+(0.03339420027007282)*u**4+(-0.01395440267855593)*u**3+(-0.0007800178686230538)*u**5+(5.291361442959047e-06)*u**6
    # f = (1.5457295271224543)*u+(-0.9620961709193429)*u**3+(0.3557495661427541)*1+(0.24965568742029573)*u**4+(0.16198516925539228)*u**2+(-0.00486351191446919)*u**5+(3.229259076735189e-05)*u**6
    f = (1.749363429305933)*u+(-0.6412202270470834)*u**2+(0.42604598861290155)*1+(0.03422744421291603)*u**4+(-0.03333629959294421)*u**3+(-0.0012964043575840343)*u**5+(1.6239406015018106e-05)*u**6
    return f

def plot_real_and_predict_function(name='name'):
    x = []
    N = 100
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    f_real = np.array([f_real_function(ele) for ele in x])
    f_predict = np.array([f_predict_function(ele) for ele in x])

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict - (f_predict[0] - f_real[0]), label='fix', color='orange', linestyle='--', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('f(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2, plot_3],
               labels=['exact', 'prediction', 'revised prediction'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'real_prediction_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


# 画出真实和噪声数据
def plot_fixed_time_step_real_noise_data(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_noise_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2 = plt.scatter(x_label, fix_timestep_noise_data, c="black")
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=['clean', 'noise'],
               loc="upper right", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' + 'exact_noise_data'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_noise_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_observation_distribution(time_points, name):
    all_data = np.load('data/' + name + '_U.npy')
    time_points = time_points
    data = all_data[time_points, :, :].flatten()
    new_data = []
    # for ele in data:
    #     if ele != 0.0 and ele != 1.0:   #and ele != 0.8 and ele != 0.6 and ele != 0.4 and ele != 0.3 and ele != 0.7:
    #         new_data.append(ele)
    new_data = data
    weights = [1./len(new_data)] * len(new_data)
    label_fontsize = 50
    ticks_fontsize = 50
    linewith_frame = 4
    fig = plt.figure(figsize=(40, 20))
    # plt.hist(new_data, weights=weights, bins=50)
    plt.hist(new_data, bins=50)
    plt.ylim(0, 200.0)
    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('proportion', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    # major_ticks_top = np.linspace(0, 1, 21)
    # ax1.set_yticks(major_ticks_top)
    plt.grid(linewidth=0.2)
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_observation_distribution'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'observation_distribution_sub' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_init_states(data, example_index, name):
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    init_states = data[0]
    init_state = init_states[example_index]
    x_label = []
    dx = 10/200   # 0.05
    for i in range(200):
        x_label.append(i * dx)
    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    plot_1, = plt.plot(x_label, init_state, label='observation', color='blue', linestyle='-', linewidth=linewith)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim(-0.1, 1.1)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_initial_states'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'initial_state_data' + '_' + 'example' + '_' + str(example_index) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)







if __name__ == "__main__":
    # test_data_name = 'N_200_example_1_dt_0.1_layer_10_beta_10'
    # test_data_name = 'N_200_example_1_dt_0.1_layer_10_beta_10_extra_test'
    # test_data_name = 'N_200_example_3_dt_0.1_layer_10_beta_10_noise_0.05'
    # test_data_name = 'N_200_example_1_dt_0.1_layer_10_beta_10_noise_0.03_1'
    test_data_name = 'N_200_example_3_dt_0.1_layer_10_beta_10_noise_0.05'

    # time_steps_real = 42  # 84   # 42
    # 不含有噪声数据的
    # time_steps_predict = 85  # 43
    # 含有噪声数据的
    # time_steps_predict = 49  # 98  # 49(noise= 0.05 example 1)
    # time_steps_predict = 53  # 106  # 53 (noise= 0.05 example 2)
    # time_steps_predict = 84  # 84  # 42  (noise= 0.05 example 3)
    # time_steps_predict = 42  # 42  # 85 (noise= 0.03)
    # time_steps_predict = 86  # 43  # 86 (第二种 noise= 0.03)
    # time_steps_predict = 50  # 100  # 50  (noise= 0.05 example 3, 利用了五噪声时候的初始值)

    # cell_numbers = 200
    # example_index = 0
    # time = 1
    # real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name +'.npy'

    # 下载数据
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)


    # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # 不含有噪声
    # plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, cell_numbers, name=test_data_name)
    # 含有噪声
    # real_no_noise_data = np.load('data/N_200_example_3_dt_0.1_layer_10_beta_10_U' + '.npy')
    # plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, time_steps_real, time_steps_predict, time, example_index, cell_numbers, name=test_data_name)


    # 真实函数和预测函数
    # plot_real_and_predict_function(name=test_data_name)


    # # 画出观测数据得分布图
    time_point_list = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38]
    plot_observation_distribution(time_points=time_point_list, name=test_data_name)


    # 画出噪音数据下的初始状态
    # example_index = 1
    # plot_init_states(real_data, example_index, test_data_name)


    # 比较真实数据和加了误差的数据
    # time_steps_real = 13   # 13, 26, 38
    # time_steps_predict = time_steps_real
    # time = 0.3
    # example_index = 0
    # plot_fixed_time_step_real_noise_data(real_no_noise_data, real_data, time_steps_real, time_steps_predict, time, example_index, name=test_data_name)






