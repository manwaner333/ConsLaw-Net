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


def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time,  example_index, cell_numbers, name):

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
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * (-100) * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f
def f_predict_function(u):
    # f = (-3.994657753009525)*u**2+(3.8191418764266465)*u+(1.2354160050061043)*u**3+(0.17436684167576308)*1+(-0.07021102101487424)*u**4+(-0.006222562944701795)*u**5+(0.0006823684602326993)*u**6+(-2.0573007615028174e-05)*u**7
    # f = (-4.1761141240201605)*u**2+(3.8610994359518105)*u+(1.3172571300896032)*u**3+(0.1722597180940788)*1+(-0.022446676490739198)*u**4+(-0.01089903558120832)*u**5+(0.0007593531435613084)*u**6+(-1.817756958204667e-05)*u**7
    # f = (-4.1861754006312335)*u**2+(3.8628830813424324)*u+(1.1986633920666265)*u**3+(0.3272760198992488)*1+(0.05720039671457576)*u**4+(-0.020085695427667055)*u**5+(0.0011554586277540686)*u**6+(-2.486156137760491e-05)*u**7
    # f = (-11.447205511210075)*u**3+(5.110191138028528)*u**2+(2.4242399330610347)*u**5+(2.356354824143221)*u**4+(2.1378076974774003)*u+(0.4483317879123521)*u**6+(0.41789557721150394)*1+(0.03223973310324886)*u**7+(0.0008161768212000247)*u**8
    # f = (1.1805458962124218)*u+(0.7039623449350774)*1+(-0.1892554393137135)*u**2+(0.04171720031063284)*u**3+(0.0029196626482811197)*u**4+(-0.00020179301113928077)*u**5+(3.2403255831329447e-06)*u**6
    # f = (0.7115053203845328)*u+(0.5297219412361184)*1+(0.2904120696983826)*u**2+(0.17938121620828182)*u**3+(0.019008177041830013)*u**4+(0.0006360436010996981)*u**5+(4.1987800581546595e-06)*u**6
    # f = (1.217292131636691)*u+(0.7004749675456572)*1+(-0.18429887692148608)*u**2+(0.04827494594480958)*u**3+(0.002380613687370704)*u**4+(-0.0003433934580337105)*u**5+(8.894164423860896e-06)*u**6
    # f = (1.2341274190735039)*u+(0.4775756342137945)*1+(-0.00445622491114011)*u**2+(0.00017362044761963286)*u**3+(-4.883624747160988e-06)*u**4
    # f = (1.1643049406491504)*u+(0.4828986557076078)*1+(-0.003492882710417564)*u**2+(0.0001647979256822399)*u**3+(-5.7456580645744326e-06)*u**4
    f = (7.902438954865842)*u**3+(-5.9867848414950835)*u**4+(-2.8596064344636365)*u**2+(1.5200190278723675)*u**8+(1.365877594603687)*u**5+(1.0464729259352052)*u+(-0.9365830691669547)*u**7+(-0.7911292750366253)*1+(-0.5393935793166017)*u**10+(-0.5090870187420392)*u**6+(0.4144551823866562)*u**11+(-0.35903049820638944)*u**9+(-0.14405232353335323)*u**13+(0.032559509003424476)*u**12+(0.02978593618748953)*u**14+(0.02684671538864626)*u**15+(-0.012485648378272134)*u**16+(0.002738758071935589)*u**18+(-0.0024795899134887984)*u**17+(-0.0003875211247811535)*u**20+(4.447551807019916e-05)*u**19+(3.2677743208893346e-05)*u**22+(1.0427104408793159e-05)*u**21+(-1.3588683675434044e-06)*u**24
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
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'real_prediction_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    test_data_name = 'N_200_example_5_dt_0.1_layer_10_beta_negative_100_adjust_du'  # _no_stableloss

    time_steps_real = 174
    time_steps_predict = 100  # 64, 128
    cell_numbers = 200
    example_index = 1
    time = 2
    real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    predict_data_file = 'data/' + 'predict_' + test_data_name + '_U_cell_' + str(cell_numbers) +'.npy'

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
    # plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time,  example_index, cell_numbers, name=test_data_name)

    # 真实函数和预测函数
    plot_real_and_predict_function(name=test_data_name)


