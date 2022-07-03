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


def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index,cell_numbers, name):

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
    dx = 10/200   # 0.05
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
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * (-300) * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f
def f_predict_function(u):
    # f = (1.6011401235156726)*u**2+(-0.8551301887148467)*1+(-0.8232180858722162)*u+(0.6225849799091898)*u**3+(-0.06627620338116746)*u**4+(-0.022964316846394894)*u**5+(-0.0008145840085516209)*u**7+(-0.00040735270425778234)*u**6+(-0.0001736409354429067)*u**8+(4.821804066764993e-06)*u**9+(3.6066068243705745e-06)*u**10
    # f = (3.144519499656731)*u**2+(-1.8635202815137324)*u+(1.1426723430178007)*u**3+(-0.844252514401015)*1+(-0.7710810893321832)*u**6+(-0.6990245498496053)*u**5+(0.6535612346377152)*u**4+(-0.5602705386591901)*u**7+(-0.17634773307449428)*u**8+(0.01758606390364136)*u**9+(-0.0006946490089537006)*u**10+(1.5374900817787597e-05)*u**11
    # f = (5.401938330919747)*u**2+(-2.537104522265853)*u+(-1.51286564812682)*u**4+(-0.9267822247791858)*1+(-0.5058075445133869)*u**6+(0.16334039249112928)*u**5+(0.03179228606607515)*u**7+(-0.0268396514556517)*u**8+(0.02029711414640477)*u**3+(0.0011809832774636714)*u**9+(-2.2338237166676165e-05)*u**10
    # f = (3.363669580226715)*u**2+(-1.9258240092825858)*1+(-1.5977054830618875)*u+(-1.3679615612609792)*u**6+(0.9027809619414652)*u**4+(-0.5445502557778902)*u**8+(-0.45419812026725315)*u**3+(0.38511443153286307)*u**7+(0.3338654328141775)*u**5+(0.09551898209686706)*u**9+(-0.007137353481296685)*u**10+(0.0002832256915137206)*u**11+(-6.3048821450576816e-06)*u**12
    # f = (3.144519499656731)*u**2+(-1.8635202815137324)*u+(1.1426723430178007)*u**3+(-0.844252514401015)*1+(-0.7710810893321832)*u**6+(-0.6990245498496053)*u**5+(0.6535612346377152)*u**4+(-0.5602705386591901)*u**7+(-0.17634773307449428)*u**8+(0.01758606390364136)*u**9+(-0.0006946490089537006)*u**10+(1.5374900817787597e-05)*u**11
    # f = (5.134048092413524)*u**2+(-2.553722794998459)*u+(-1.8314136826654495)*u**4+(-0.9536871519925976)*1+(0.8215571683261532)*u**3+(-0.41872775277374763)*u**6+(-0.07546583929470191)*u**5+(-0.029028898722468575)*u**7+(-0.02307282519166819)*u**8+(-0.0015081898951079791)*u**9+(1.537363283854864e-05)*u**10+(3.433792078464884e-06)*u**11
    f = (5.001537748702942)*u**2+(3.8256314816454196)*u**7+(-3.814492966772781)*u**8+(-2.9502226685643897)*u**4+(2.919704530921351)*u**9+(-2.3809049002826366)*u**10+(-2.26366376774374)*u**6+(-2.1971426014500053)*u+(1.9289515407315003)*u**11+(1.6142011548320496)*u**5+(-1.2896989524054376)*u**12+(0.7549378989821096)*u**13+(-0.5780906864346561)*1+(-0.44967432692335607)*u**14+(0.27901703593448657)*u**3+(0.2565285213755303)*u**15+(-0.12267019919138683)*u**16+(0.05357031967386851)*u**17+(-0.024439297223805407)*u**18+(0.009929265066243009)*u**19+(-0.0032021468430948634)*u**20+(0.0010156919048501596)*u**21+(-0.00033309802397237865)*u**22+(7.902038943942244e-05)*u**23+(-1.1747522404057791e-05)*u**24+(1.4863423885086766e-06)*u**25
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
    # #  保存
    # save_dir = 'figures/' + name + '_function'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = '/' + 'real_prediction_function' + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    test_data_name = 'N_200_example_6_dt_0.1_layer_12_beta_negative_300'  # _no_stableloss

    time_steps_real = 272  # 363
    time_steps_predict = 367   # 489
    cell_numbers = 200
    example_index = 5
    time = 1  # 实际时间是1.5
    real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    predict_data_file = 'data/' + 'predict_' + test_data_name +'.npy'

    # time_steps = 599    # 149     # 299   # 449  # 599
    # cell_numbers = 1600
    # example_index = 0
    # real_data_file = 'data/' + test_data_name + '_U_cell_' + str(cell_numbers) + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name + '_U_cell_' + str(cell_numbers) +'.npy'

    # 下载数据
    real_data = np.load(real_data_file)
    predict_data = np.load(predict_data_file)


    # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, cell_numbers, name=test_data_name)

    # 真实函数和预测函数
    plot_real_and_predict_function(name=test_data_name)


