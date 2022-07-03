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
sns.set_theme()

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
    fig, axes = plt.subplots(1, 3, figsize=(50, 16), sharey=True)
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
    g1.set_title("observation u(x, t)", fontsize=title_fontsize)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=cbar_size)


    g2 = sns.heatmap(np.flip(predict_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, ax=axes[1])
    g2.set_ylabel('T', fontsize=label_fontsize)
    g2.set_xlabel('X', fontsize=label_fontsize)
    g2.set_xticklabels([])
    g2.set_yticklabels([])
    g2.set_title("prediction u(x, t)", fontsize=title_fontsize)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=cbar_size)

    g3 = sns.heatmap(np.flip(real_data[:, example_index, :], 0) - np.flip(predict_data[:, example_index, :], 0)
                     , cmap="YlGnBu", cbar=True, ax=axes[2])
    g3.set_ylabel('T', fontsize=label_fontsize)
    g3.set_xlabel('X', fontsize=label_fontsize)
    g3.set_xticklabels([])
    g3.set_yticklabels([])
    g3.set_title("error u(x, t)", fontsize=title_fontsize)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=cbar_size)

    # plt.show()

    save_dir = 'figures/' + name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/real_predict_loss_example' + '_' + str(example_index) + '_' + '.pdf'
    fig.savefig(save_dir + file_name)


def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time,  example_index,cell_numbers, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(50, 20))
    plt.plot()
    x_label = []
    dx = 10/1600   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='orange', linestyle='--', linewidth=linewith)
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
               labels=['observation', 'prediction'],
               loc="upper left", fontsize=50,)
    plt.show()

    save_dir = 'figures/' + name + '_' +'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_predict_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def f_real_function(u):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * (200) * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f
def f_predict_function(u):
    f = (-3.994657753009525)*u**2+(3.8191418764266465)*u+(1.2354160050061043)*u**3+(0.17436684167576308)*1+(-0.07021102101487424)*u**4+(-0.006222562944701795)*u**5+(0.0006823684602326993)*u**6+(-2.0573007615028174e-05)*u**7
    return f

def plot_real_and_predict_function(name='name'):
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    f_real = np.array([f_real_function(ele) for ele in x])
    f_predict = np.array([f_predict_function(ele) for ele in x])

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict - (f_predict[0] - f_real[0]), label='fix', color='orange', linestyle='--', linewidth=linewith)

    # plt.title(label='X', fontsize=title_fontsize)
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
               labels=['observation', 'prediction', 'revised prediction'],
               loc="upper left", fontsize=50,)
    plt.show()
    # # #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'real_prediction_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    test_data_name = 'N_200_example_1_dt_0.1_layer_10_beta_200'  # _no_stableloss

    # time_steps_real = 64   # 64, 128
    # time_steps_predict = 64  # 64, 128
    # cell_numbers = 200
    # example_index = 0
    # time = 1
    # real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name + '_U_cell_' + str(cell_numbers) +'.npy'

    # time_steps = 599    # 149     # 299   # 449  # 599
    # cell_numbers = 1600
    # example_index = 0
    # real_data_file = 'data/' + test_data_name + '_U_cell_' + str(cell_numbers) + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name + '_U_cell_' + str(cell_numbers) +'.npy'

    # 下载数据
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)


    # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time,  example_index, cell_numbers, name=test_data_name)

    # 真实函数和预测函数
    plot_real_and_predict_function(name=test_data_name)


