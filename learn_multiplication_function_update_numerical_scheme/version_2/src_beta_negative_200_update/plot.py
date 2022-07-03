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

def f_real_function(u):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * (-200) * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f
def f_predict_function(u):
    # f = (10.870569543777721)*u**3+(-10.75802406468404)*u**5+(-6.357706418861304)*u**2+(5.985472148457635)*u**4+(-4.981157650859255)*u**6+(4.033159839002847)*u**7+(0.8962432176370594)*u+(0.8686058745131906)*u**8+(-0.6211636210715211)*1+(0.5814181306662216)*u**10+(-0.4223469813775886)*u**9+(0.3569887623517616)*u**11+(-0.09368646627676888)*u**12+(-0.009397168953653003)*u**15+(0.008230389794564549)*u**13+(0.006662143366984371)*u**14+(-0.0020967299163763644)*u**16
    f = (1.5464196468395395)*u**3+(0.8439764834701333)*u**2+(-0.804942215121792)*1+(-0.7321282232795286)*u**6+(0.524687827899892)*u**4+(-0.40928828691757596)*u**5+(-0.3847272300433369)*u**7+(-0.36865412853164836)*u+(-0.10209402338637667)*u**8+(0.009019125417110606)*u**10+(0.008078770995170632)*u**9+(0.002541827816889442)*u**11+(0.0002641590880145429)*u**12+(3.7202151754650344e-05)*u**13+(3.2252379372936605e-05)*u**14+(8.415858760045225e-06)*u**15
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
    # ax1.set_facecolor('w')
    plt.show()
    # # #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'real_prediction_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_observation_distribution(time_points, name):
    all_data = np.load('data/' + name + '_U.npy')
    time_points = time_points
    data = all_data[time_points, :, :].flatten()
    # new_data = []
    # for ele in data:
    #     if ele != 0.0 and ele != 1.0 and ele != 0.8 and ele != 0.6 and ele != 0.4 and ele != 0.3 and ele != 0.7:
    #         new_data.append(ele)
    new_data = data
    weights = [1./len(new_data)] * len(new_data)
    label_fontsize = 50
    ticks_fontsize = 50
    linewith_frame = 4
    fig = plt.figure(figsize=(40, 20))
    # plt.hist(new_data, weights=weights, bins=50)
    plt.hist(new_data, bins=50)
    plt.ylim(0, 200)
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
    # # #  保存
    save_dir = 'figures/' + name + '_observation_distribution'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'observation_distribution_sub' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)




if __name__ == "__main__":
    test_data_name = 'N_200_example_6_dt_0.1_layer_10_beta_negative_200'

    # time_steps_real = 202   # 269 , 202
    # time_steps_predict = 292   # 389, 292 example=6
    # time_steps_predict = 241   # 322, 241,   example=2
    # cell_numbers = 200
    # example_index = 0
    # time = 1  # 实际1是1.5
    # real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name +'.npy'
    #
    # # 下载数据
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)


    # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, cell_numbers, name=test_data_name)

    # 真实函数和预测函数
    # plot_real_and_predict_function(name=test_data_name)

    # 画出观测数据得分布图
    time_point_list = [0, 14, 27, 41, 54, 68, 81, 95, 108, 122, 135, 149]
    plot_observation_distribution(time_points=time_point_list, name=test_data_name)





