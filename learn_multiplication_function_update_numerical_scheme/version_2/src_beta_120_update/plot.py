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
import re
np.random.seed(0)

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
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * 120 * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f
def f_predict_function(u):
    # f = (-6.834229332266385)*u**3+(2.646872081621761)*u**4+(2.1957148340849746)*u**2+(2.1770218776442927)*u+(0.7682667392176022)*u**5+(0.4030515706146376)*1+(0.0648329581342991)*u**6+(0.0022488876791136214)*u**7+(2.792806935589932e-05)*u**8
    # f = (2.9195045787753653)*u+(-2.2289553896296996)*u**2+(0.3998115127235462)*1+(0.18310164373513982)*u**3+(0.08692487407504992)*u**4+(0.002690562593069084)*u**5+(2.8010732216107646e-05)*u**6
    # f = (3.031126833243591)*u+(-2.545803598226914)*u**2+(0.3908824488421731)*u**3+(0.34962494214734946)*1+(0.1548342539397727)*u**4+(-0.03895082618140552)*u**5+(0.0030478843876559575)*u**6+(-9.698342247538217e-05)*u**7+(1.086007327819461e-06)*u**8
    # f = (3.035114909622242)*u+(-2.6228134282381155)*u**2+(0.4237774918587127)*u**3+(0.4165985106038784)*1+(0.2126117128031912)*u**4+(-0.039346205048192084)*u**5+(0.002338176733472871)*u**6+(-5.637267739073105e-05)*u**7
    # f = (21.79634900111228)*u**4+(-21.10548976860509)*u**3+(-10.301214750245114)*u**5+(6.787527847462155)*u**2+(2.4417065138644864)*u**6+(1.6637591333546562)*u+(0.44271297060579684)*1+(-0.2826501690527755)*u**7+(0.01271822023951458)*u**8
    # f = (21.7963)*u**4+(-21.1055)*u**3+(-10.3012)*u**5+(6.7875)*u**2+(2.4417)*u**6+(1.6638)*u+(0.4427)*1+(-0.2827)*u**7+(0.0127)*u**8
    f = 1.5 * u + (0.068*120-0.5612) * u**2 + (-0.21*120 + 1.056) * u**3 + (0.234*120 - 1.89) * u**4  \
        + (-0.1228 * 120 + 1.049) * u**5 + (0.03 * 120 - 0.24) * u**6
    # f = (10.122683182315917)*u**4+(-9.98090160196883)*u**3+(-3.930645207921477)*u**5+(2.934910655866976)*u+(1.2992072383029818)*u**2+(0.7115993492737755)*u**6+(0.24057521136522916)*1+(-0.06040612686168796)*u**7+(0.001941864449732949)*u**8
    # f = (2.821571680728443)*u+(-1.7376267832912273)*u**2+(0.4262494394842354)*1+(-0.35359900691794455)*u**3+(0.11832523245807955)*u**4+(0.026803955907172085)*u**5+(0.0016280839286000666)*u**6+(3.265946913207999e-05)*u**7
    # f = (-4.921324843273608)*u**3+(2.376138260382418)*u+(1.7820253935769925)*u**4+(1.0586948612603395)*u**2+(0.6507773456107627)*u**5+(0.5100653529199363)*1+(0.06531203122984827)*u**6+(0.0026593134080233776)*u**7+(3.850552062478322e-05)*u**8
    # f = (2.992355219210167)*u+(-1.9020609801302528)*u**2+(-1.0432818027918314)*u**3+(0.9300084646760871)*u**4+(0.5132954627216219)*1+(0.030124267816086295)*u**5+(0.0003433005384400041)*u**6+(1.6373365855913076e-06)*u**7
    # f = (-12.521285420210774)*u**3+(10.446058050098241)*u**4+(3.9880969958859342)*u**2+(-3.3615151565836796)*u**5+(1.9654407032375663)*u+(0.5020271517005006)*u**6+(0.3695253164367981)*1+(-0.035135628409731175)*u**7+(0.0009326630269957753)*u**8
    # f = (-5.003773209557326)*u**3+(2.280676828553506)*u+(1.3342259700172239)*u**2+(1.266458844986153)*u**4+(0.9805842451937651)*u**5+(0.45264255197250686)*1+(0.1583734672062787)*u**6+(0.00998955932629509)*u**7+(0.00022096311630228688)*u**8
    # f = (2.838449581560666)*u+(-2.0062805568206867)*u**2+(0.4038836462983143)*1+(0.07653946992822065)*u**4+(0.00917318807290743)*u**3+(0.006895078135572286)*u**5+(0.00019958150048030254)*u**6+(1.8199340085797907e-06)*u**7
    # f = (2.915170940718388)*u+(-2.4205442878297534)*u**2+(0.4071528314050516)*u**3+(0.39214206272273766)*1+(0.12685622984162667)*u**4+(-0.03301458026984626)*u**5+(0.00255578323933569)*u**6+(-7.92899745332095e-05)*u**7
    # f = (24.629216795317745)*u**4+(-23.425112671798402)*u**3+(-12.110430837865726)*u**5+(7.709845229662649)*u**2+(3.021617990809614)*u**6+(1.5297492322870434)*u+(0.452032985130408)*1+(-0.3701770912083237)*u**7+(0.017666000727127224)*u**8
    # f = (-4.350013527033505)*u**3+(3.0270331053370514)*u**4+(2.5648060493302953)*u+(0.40690931666141184)*1+(-0.30411338943945443)*u**5+(0.050380909862039164)*u**2+(0.011904321897183151)*u**6+(-0.00020389755617863776)*u**7+(1.2789208494823847e-06)*u**8
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
    # #  保存
    # save_dir = 'figures/' + name + '_function'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = '/' + 'real_prediction_function' + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_observation_distribution(time_points, name):
    all_data = np.load('data/' + name + '_U.npy')
    time_points = time_points
    data = all_data[time_points, :, :].flatten()
    new_data = []
    for ele in data:
        if ele != 0.0 and ele != 1.0 and ele != 0.9:
            new_data.append(ele)
    # new_data = data
    weights = [1./len(new_data)] * len(new_data)
    label_fontsize = 50
    ticks_fontsize = 50
    linewith_frame = 4
    fig = plt.figure(figsize=(40, 20))
    # plt.hist(new_data, weights=weights, bins=50)
    plt.hist(new_data, bins=50)
    # plt.ylim(0, 200)
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
    file_name = '/' + 'observation_distribution_all' + '.pdf'
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
               labels=['exact', 'noise'],
               loc="upper right", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' +'exact_noise_data'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_noise_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_loss_function_all(name):

    f = open("checkpoint/" + name + "/" + "loss.txt", encoding="utf8")
    line = f.readline()
    value_list = []
    while line:
        if "data loss0" in line and "nan" not in line:
            # print(line)
            number = re.findall(r'\d+(?:\.\d+)?', line)
            # print(number)
            value_list.append(float(number[1]))
        line = f.readline()
    x = np.arange(0, len(value_list))
    linewith = 5
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    plot_1, = plt.plot(x, value_list, label='loss', color='red', linestyle='-', linewidth=linewith)
    plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)

    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_loss'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'loss_function_all' + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_loss_function_sub(name, index, limit):

    f = open("checkpoint/" + name + "/" + "loss.txt", encoding="utf8")
    line = f.readline()
    value_list = []
    while line:
        if "data loss0" in line and "nan" not in line:
            # print(line)
            number = re.findall(r'\d+(?:\.\d+)?', line)
            # print(number)
            value_list.append(float(number[1]))
        line = f.readline()
    linewith = 2
    linewith_frame = 1
    label_fontsize = 10
    ticks_fontsize = 10

    new_value_list = []
    for i in range(len(value_list)):
        if i >= index:
            new_value_list.append(value_list[i])
    x = np.arange(0, len(new_value_list))
    fig = plt.figure(figsize=(8, 5))
    plot_1, = plt.plot(x, new_value_list, label='loss', color='green', linestyle='-', linewidth=linewith)
    # plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlim(0, len(new_value_list))
    plt.ylim(0.0, limit)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)

    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_loss'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'loss_function_sub' + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)





if __name__ == "__main__":
    # test_data_name = 'N_200_example_4_dt_0.1_layer_10_beta_120_1_noise_0.05'
    # test_data_name = 'N_200_example_1_dt_0.1_layer_10_beta_120_extra_test'
    # test_data_name = 'N_200_example_2_dt_0.1_layer_10_beta_120_noise_0.05'
    # test_data_name = 'N_200_example_2_dt_0.1_layer_10_beta_120_extra_test'
    # test_data_name = 'N_200_example_4_dt_0.1_layer_10_beta_120_noise_0.05'
    test_data_name = 'N_200_example_1_dt_0.1_layer_10_beta_120_2'

    # time_steps_real = 69  # 69  # 138
    # time_steps_predict = 68  # 68   # 136
    # time_steps_predict = 66  # 66   # 132    # noise=0.05 and example=2
    # time_steps_predict = 89  # 89   # 178    # example=1 盒子形状的初始值
    # time_steps_predict = 69  # 69   # 138    # noise=0.05 and example=4

    # cell_numbers = 200
    # example_index = 3
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
    # real_no_noise_data = np.load('data/N_200_example_4_dt_0.1_layer_10_beta_120_U' + '.npy')
    # plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, time_steps_real, time_steps_predict, time, example_index, cell_numbers, name=test_data_name)


    # 真实函数和预测函数
    plot_real_and_predict_function(name=test_data_name)

    # 画出观测数据得分布图
    # time_point_list = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
    # plot_observation_distribution(time_points=time_point_list, name=test_data_name)

    # 比较真实数据和加了误差的数据
    # time_steps_real = 21  # 21, 42, 63
    # time_steps_predict = time_steps_real
    # time = 0.3
    # example_index = 0
    # plot_fixed_time_step_real_noise_data(real_no_noise_data, real_data, time_steps_real, time_steps_predict, time, example_index, name=test_data_name)

    # 画出损失函数
    # plot_loss_function_all(name=test_data_name)
    # index = 40
    # limit = 0.001
    # plot_loss_function_sub(name=test_data_name, index=index, limit=limit)





