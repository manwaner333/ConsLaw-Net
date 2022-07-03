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

    plt.show()

    save_dir = 'figures/' + name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/real_predict_loss_example' + '_' + str(example_index) + '_' + '.pdf'
    fig.savefig(save_dir + file_name)

def plot_fixed_time_step(real_data, predict_data, real_time_step, predict_time_step, time, N, example_index, name):

    fix_timestep_real_data = real_data[real_time_step, example_index, :]
    fix_timestep_predict_data = predict_data[predict_time_step, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(50, 20))
    plt.plot()
    x_label = []
    dx = 10/N
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
    f = np.power(u, 2) / (np.power(u, 2) + 1.5 * np.power(1 - u, 2))
    return f

def f_predict_function(u):
    # return np.power(u, 2) / (np.power(u, 2) + 0.5 * np.power(1 - u, 2))
    # molecular = (-0.7229038444402525)*1+(0.30923247703582196)*u+(-0.07609330155108462)*u**2+(0.013612174497125163)*u**3+(0.0036661732444398278)*u**4
    # denominator =(1.4825916471954779)*u+(-1.455440508455919)*u**2+(-0.7606836407412769)*1+(0.08352557751284082)*u**3+(0.02249598236994642)*u**4
    molecular = (-0.9849056089941699)*u+(0.11416894974399465)*1+(-0.01787824205675676)*u**2+(-0.0004935280651586317)*u**4+(-0.00040611000485279563)*u**3
    denominator = (-1.4792209811042405)*1+(1.086811451691851)*u+(-0.28251742423938087)*u**2+(0.003035202676795784)*u**4+(0.0024975807067963984)*u**3
    return molecular/denominator


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
    plot_3, = plt.plot(x, f_predict - (f_predict[0] - f_real[0]), label='revised prediction', color='orange', linestyle='--', linewidth=linewith)

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
    # plt.grid(b=False)
    plt.show()
    # # # #  保存
    # save_dir = 'figures/' + name + '_function'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = '/' + 'real_prediction_function' + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    # N = 400
    test_data_name = 'N_400_example_1_dt_0.6_layer_15_beta_5'
    # real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name + '_U' + '.npy'
    # real_time_step = 58  # 58  # 117   # 176   # 235
    # predict_time_step = 58
    # time = 10
    # example_index = 0
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)

    # N = 1600
    # test_data_name = 'N_1600_example_1_dt_0.6_layer_20_beta_0.5'
    # real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name + '_U' + '.npy'
    # real_time_step = 704  # 234  # 469   # 704   # 940
    # predict_time_step = 704
    # time = 30
    # example_index = 0
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)


    # # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)
    # # 固定时间下真实和预测情况
    # plot_fixed_time_step(real_data, predict_data, real_time_step, predict_time_step, time, N, example_index, name=test_data_name)
    # f(u)函数的差别
    plot_real_and_predict_function(name=test_data_name)


