import threading
import time
import os
import h5py
import copy
import joblib
import numpy as np
import numpy.fft
import pandas as pd
import networkx as nx
import scipy
import torch
import torch_geometric
import tqdm
from matplotlib import pyplot as plt, gridspec
from sklearn import preprocessing
from torch.utils.data import TensorDataset, Dataset
from torch_geometric import utils
from torch_geometric.data import Data
from scipy.signal import savgol_filter
from torch_geometric.loader import DataLoader
from NetworkClass import PreModel
import math
import pywt
from pyvis.network import Network
import matplotlib
from dtw import *


class PreMyDataset(Dataset):
    def __init__(self, input_data, output_data, device):
        super(PreMyDataset, self).__init__()
        self.w = torch.from_numpy(input_data).to(torch.float32).to(device)
        self.x = torch.from_numpy(output_data).to(torch.float32).to(device)

    def __len__(self):
        assert self.w.shape[0] == self.x.shape[0]
        length = self.w.shape[0]
        return length

    def __getitem__(self, item):
        w = self.w[item]
        x = self.x[item]
        return w, x


def sample(data, frequency):
    """
    采样函数
    :param data: 待采样数据
    :param frequency: 采样频率
    :return: 采样完后的数据集
    """
    rate = frequency
    subset = data[::rate]
    print(subset.shape)
    return subset


def load_all_data_to_pd(path, dataset_list, sample_value, optimize_label=False):
    """
    数据载入函数，按指定载入所需要的数据集 (DS01-DS08，中的一个或多个，乃至全部)，将文件列表中的所需数据集转换成 pd 格式的数据
    :param path: 数据集文件夹所在路径
    :param dataset_list: 需要载入的数据集列表 (DS01-DS08)
    :param sample_value: 数据的采样频率
    :param optimize_label: 对于剩余寿命标签的优化，将原有的线性标签变化为非线性标签 (以健康与否为标准分割，健康时寿命为恒值，非健康线性衰退)
    :return: 返回所需的数据集，格式为 pd
    """
    print("载入所有数据中......")
    names = os.listdir(path)
    print("-------分隔符--------")
    for i, subset in enumerate(dataset_list):
        data_selection = [s for s in names if subset in s]
        data_filename = path + str(data_selection[0])
        print(data_filename)
        if i == 0:
            df_data = load_raw_data(data_filename, sample_value=sample_value, optimize_label=optimize_label)
            df_data['set'] = subset
        else:
            df_temp = load_raw_data(data_filename, sample_value=sample_value, optimize_label=optimize_label)
            df_temp['set'] = subset
            df_data = pd.concat([df_data, df_temp], axis=0)

    df_data.reset_index(drop=True, inplace=True)

    return df_data


def load_raw_data(dataset_filepath, sample_value=None, optimize_label=False, hs_onehot=False, add_xv=False):
    """
    载入指定的数据集内容，DS01-DS08 中的一个
    :param dataset_filepath: 数据集路径
    :param sample_value: 采样频率
    :param optimize_label: 对于剩余寿命标签的优化，将原有的线性标签变化为非线性标签 (以健康与否为标准分割，健康时寿命为恒值，非健康线性衰退)
                           且将剩余寿命标签从 100-0 归一化为 1-0
    :param hs_onehot: 将健康参数标签从转换为独热码
    :param add_xv: 是否加入不可测传感器变量，默认为 False
    :return: 返回所需的数据集内容，格式为 pd
    """
    # Load data 加载原始数据集
    with h5py.File(dataset_filepath, 'r') as hdf:
        # Training set 获取训练集数据
        w_dev = np.array(hdf.get('W_dev'))  # W ——环境描述（4个：海拔、马赫数、油门、涡扇温度）
        x_s_dev = np.array(hdf.get('X_s_dev'))  # X_s ——可观测的传感器数据（14个，多一个P2，顺序与说明文件也不一致，详见学习笔记）
        x_v_dev = np.array(hdf.get('X_v_dev'))  # X_v ——不可观测的传感器估计值数据（14个，比说明文档少4类数据，详见学习笔记）
        t_dev = np.array(hdf.get('T_dev'))  # T ——健康参数(见说明文件，顺序一质)
        y_dev = np.array(hdf.get('Y_dev'))  # RUL ——发动机寿命剩余
        a_dev = np.array(hdf.get('A_dev'))  # Auxiliary ——辅助变量:数据集单元名(unit);飞行周期(cycle);飞行类型(Fc);健康状态(hs)

        # Test set 获取测试集
        w_test = np.array(hdf.get('W_test'))  # W
        x_s_test = np.array(hdf.get('X_s_test'))  # X_s
        x_v_test = np.array(hdf.get('X_v_test'))  # X_v
        t_test = np.array(hdf.get('T_test'))  # T
        y_test = np.array(hdf.get('Y_test'))  # RUL
        a_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Variable name 获取变量名
        w_var = np.array(hdf.get('W_var'))
        x_s_var = np.array(hdf.get('X_s_var'))
        x_v_var = np.array(hdf.get('X_v_var'))
        t_var = np.array(hdf.get('T_var'))
        y_var = np.array(['RUL'], dtype='U20')
        a_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        w_var = list(np.array(w_var, dtype='U20'))
        x_s_var = list(np.array(x_s_var, dtype='U20'))
        x_v_var = list(np.array(x_v_var, dtype='U20'))
        t_var = list(np.array(t_var, dtype='U20'))
        y_var = list(np.array(y_var, dtype='U20'))
        a_var = list(np.array(a_var, dtype='U20'))

    df_w_dev = pd.DataFrame(data=w_dev, columns=w_var)  # flight conditions for training
    df_a_dev = pd.DataFrame(data=a_dev, columns=a_var)  # Auxiliary for training
    df_y_dev = pd.DataFrame(data=y_dev, columns=y_var)  # RUL for training
    df_t_dev = pd.DataFrame(data=t_dev, columns=t_var)  # Degradation for training

    df_w_test = pd.DataFrame(data=w_test, columns=w_var)  # flight conditions for testing
    df_a_test = pd.DataFrame(data=a_test, columns=a_var)  # Auxiliary for testing
    df_y_test = pd.DataFrame(data=y_test, columns=y_var)  # RUL for testing
    df_t_test = pd.DataFrame(data=t_test, columns=t_var)  # Degradation for testing

    print(t_var)

    # 训练集数据
    df_temp_dev = pd.concat([df_w_dev, df_a_dev], axis=1)
    df_temp_dev['index'] = df_temp_dev.index.values
    # print(df_temp_dev.shape)
    # print(df_temp_dev.head(10))

    df_xs_dev = pd.DataFrame(data=x_s_dev, columns=x_s_var)
    df_xs_dev['flag'] = 1
    df_dev = pd.concat([df_temp_dev, df_xs_dev], axis=1)
    if add_xv is True:
        df_xv_dev = pd.DataFrame(data=x_v_dev, columns=x_v_var)
        df_dev = pd.concat([df_dev, df_xv_dev], axis=1)
    # 训练集健康参数标签
    if hs_onehot is True:
        df_t_dev_temp = df_t_dev.copy()
        mask = (df_a_dev.hs == 1)
        df_t_dev_temp.loc[mask] = 0
        df_t_dev_temp[df_t_dev_temp != 0] = 1
        df_dev[df_t_dev.columns] = df_t_dev_temp.values
    else:
        df_dev[df_t_dev.columns] = df_t_dev.values
        df_dev['hs_total'] = df_dev[df_t_dev.columns].sum(axis=1)

    # 训练集剩余寿命标签
    if optimize_label is True:
        units = pd.unique(df_dev['unit'])
        df_y_dev_temp = df_y_dev.copy()
        for unit in units:
            df_unit_cycle = df_a_dev[df_a_dev.unit == unit].cycle
            df_unit_hs_cycle = df_a_dev[(df_a_dev.unit == unit) & (df_a_dev.hs == 1)].cycle
            optimize_rul = df_unit_cycle.max() - df_unit_hs_cycle.max()
            mask = (df_a_dev.unit == unit) & (df_dev.hs == 1)
            df_y_dev_temp.loc[mask] = optimize_rul
            # 再将一个单元内的标签全部归一化，也即用百分比的方式来表示寿命，健康时为 1 (100%)
            mask = (df_a_dev.unit == unit)
            df_y_dev_temp.loc[mask] = df_y_dev_temp.loc[mask].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        df_dev['RUL'] = df_y_dev_temp.values
    else:
        df_dev['RUL'] = df_y_dev.values

    print("The shape of the train data: ", df_dev.shape)

    # 测试集数据
    df_temp_test = pd.concat([df_w_test, df_a_test], axis=1)
    df_temp_test['index'] = df_temp_test.index.values

    df_xs_test = pd.DataFrame(data=x_s_test, columns=x_s_var)
    df_xs_test['flag'] = 0
    df_test = pd.concat([df_temp_test, df_xs_test], axis=1)
    if add_xv is True:
        df_xv_test = pd.DataFrame(data=x_v_test, columns=x_v_var)
        df_test = pd.concat([df_test, df_xv_test], axis=1)
    # 测试集健康参数标签
    if hs_onehot is True:
        df_t_test_temp = df_t_test.copy()
        mask = (df_a_test.hs == 1)
        df_t_test_temp.loc[mask] = 0
        df_t_test_temp[df_t_test_temp != 0] = 1
        df_test[df_t_test.columns] = df_t_test_temp.values
    else:
        df_test[df_t_test.columns] = df_t_test.values
        df_test['hs_total'] = df_test[df_t_test.columns].sum(axis=1)
    # 测试集剩余寿命标签
    if optimize_label is True:
        units = pd.unique(df_test['unit'])
        df_y_test_temp = df_y_test.copy()
        for unit in units:
            df_unit_cycle = df_a_test[df_a_test.unit == unit].cycle
            df_unit_hs_cycle = df_a_test[(df_a_test.unit == unit) & (df_a_test.hs == 1)].cycle
            optimize_rul = df_unit_cycle.max() - df_unit_hs_cycle.max()
            mask = (df_a_test.unit == unit) & (df_test.hs == 1)
            df_y_test_temp.loc[mask] = optimize_rul
            # 再将一个单元内的标签全部归一化，也即用百分比的方式来表示寿命，健康时为 1 (100%)
            mask = (df_a_test.unit == unit)
            df_y_test_temp.loc[mask] = df_y_test_temp.loc[mask].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        df_test['RUL'] = df_y_test_temp.values
    else:
        df_test['RUL'] = df_y_test.values
    print("The shape of the test data: ", df_test.shape)

    df_full = pd.concat([df_dev, df_test], axis=0)

    # 赋予时间变量[]，再采样
    units = pd.unique(df_full.unit)
    # print(units)
    df_full_copy = df_full.copy()
    df_full_copy['time'] = ''
    for unit in units:
        cycles = pd.unique(df_full[df_full.unit == unit].cycle)
        # print(cycles)
        for cycle in cycles:
            mask = (df_full.unit == unit) & (df_full.cycle == cycle)
            length = len(df_full[mask])
            time_sample = np.arange(0, length) / (length - 1)
            df_full_copy.loc[mask, 'time'] = time_sample + cycle - 1

    df_full_copy = df_full_copy.astype('float32')

    if sample_value is not None:
        data = sample(df_full_copy, sample_value)
    else:
        data = df_full_copy

    data.reset_index(drop=True, inplace=True)

    return data


def find_steady_state_data(df_data, data_var='TRA', window=100, threshold=1, picture_format='png'):
    """
    找到稳态的数据，主要以油门推拉杆角度来确定，取油门推拉杆角度 TRA 相对稳定的区间数据
    :param df_data: 原始数据，格式为 DataFrame
    :param data_var: 以该列数值变化确定稳态，默认是 油门推拉杆角度 (TRA)
    :param window: 稳态值维持长度
    :param threshold: 稳态值浮动阈值
    :param picture_format: 图片的保存格式
    :return:
    """
    # 取待检测的序列
    record_index_list = []
    # data_seq = np.array(df_data[data_var])
    data_seq = pd.DataFrame(df_data[data_var])
    slide = 1
    length = len(data_seq)
    start = data_seq.index[0]
    # start = 0
    end = start + length
    start = start + window
    for i in range(start, end, slide):
        data_seq_win = data_seq.loc[i-window:i]
        data_average = np.average(np.array(data_seq_win))
        # 判断时间窗内的数据是否在 [均值-阈值, 均值+阈值] 内，是则记录索引
        if (((data_average-threshold) <= data_seq_win.values) & (data_seq_win.values <= (data_average+threshold))).all():
            record_index_temp = np.array(data_seq_win.index)
            record_index_list.append(record_index_temp)
    record_index = np.array(record_index_list).reshape(-1)
    record_index = np.unique(record_index)
    df_data_steady_state = df_data.loc[record_index]

    # plt the data
    dataset_list = np.unique(df_data.set)
    for subset in dataset_list:
        # units = np.unique(df_data[df_data.set == subset].unit)
        units = {1}
        for unit in units:
            cycles = np.unique(df_data[(df_data.set == subset) & (df_data.unit == unit)].cycle)
            for cycle in cycles:
                df_cycle_data = df_data.loc[(df_data.set == subset) & (df_data.unit == unit) & (df_data.cycle == cycle)]
                df_cycle_steady_state = df_data_steady_state.loc[(df_data.set == subset) & (df_data.unit == unit) & (df_data.cycle == cycle)]
                plt.plot(df_cycle_data['TRA'].index, df_cycle_data['TRA'].values, c='b', label='raw data')
                plt.scatter(df_cycle_steady_state['TRA'].index, df_cycle_steady_state['TRA'].values, c='r', label='steady data')
                plt.legend()
                plt.savefig(f'./results/temp2/02/Find_steady_data_{subset}_unit{unit}_cycle{cycle}.png',
                            format=picture_format, dpi=300)
                plt.close()

    df_data_steady_state.reset_index(drop=True, inplace=True)

    return df_data_steady_state


def data_preprocess(raw_data, data_var, corr_filepath, method='spearman', threshold=0.5, device="cpu", pre_model=None,
                    scaler_file_list: list = None, graph_path: list = None, res_square=False,
                    scaled: int = None, normal_res: bool = False, picture_format='png'):
    """
    数据预处理，将原始数据转换成可以图数据化的数据集
    :param raw_data: 原始数据
    :param data_var: 需要变成图数据的变量
    :param corr_filepath: 相关变量的关系文件（即邻接矩阵），有则直接调用，无则计算生成
    :param method: 计算相似度量的方法
    :param threshold: 相关性阈值，大于等于该值认为相关，反之则不相关
    :param device: 计算设备
    :param pre_model: 预处理的模型文件，根据条件变量计算对应的传感器变量值并求残差特征
    :param scaler_file_list: 预处理归一化文件
    :param graph_path: 保存拓扑结构图
    :param res_square: 残差平方处理，默认为：False
    :param scaled: 预处理后的残差特征放大倍数，默认为：None
    :param normal_res: 将残差特征标准化处理，默认为：False
    :param picture_format: 图片保存的格式
    :return:
        df_data_scaled：可以用于图数据化的数据集
        df_correlation：邻接矩阵
        base_structure：基础图数据结构
    """

    operation_vars = ['alt', 'Mach', 'TRA', 'T2']
    sensor_vars = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    t_vars = ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
              'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
              'LPT_eff_mod', 'LPT_flow_mod']

    df_train_data, df_test_data, train_label, test_label = split_train_test(raw_data, data_var)
    print(df_train_data.keys())

    if method == 'spearman':
        # 斯皮尔曼相关性
        df_correlation = corr_analysis(df_train_data, corr_filepath, threshold=threshold, method=method)
    else:
        # 余弦相似度和高斯相似度时
        data_scaled = joblib.load(scaler_file_list[1])
        df_train_data_scaled = pd.DataFrame(data=data_scaled.transform(df_train_data), columns=df_train_data.columns)
        df_correlation = corr_analysis(df_train_data_scaled, corr_filepath, threshold=threshold, method=method)

    base_structure = graph_structure(df_correlation, data_var, graph_path, picture_format)
    print(base_structure)

    df_data_scaled = data_standardization(raw_data, data_vars=operation_vars,
                                          mode='MinMaxScaler', pre_scaler=scaler_file_list[0])
    df_data_scaled = data_standardization(df_data_scaled, data_vars=sensor_vars,
                                          mode='MinMaxScaler', pre_scaler=scaler_file_list[1])

    if pre_model is not None:
        batch_size = 128
        model = PreModel(len(operation_vars), len(sensor_vars)).to(device)
        model.load_state_dict(torch.load(pre_model))

        test_loss_list = []
        loss_fn = torch.nn.MSELoss()
        w = df_data_scaled[operation_vars]
        xs = df_data_scaled[sensor_vars]

        pre_data = PreMyDataset(np.array(w), np.array(xs), device=device)
        dataloader = DataLoader(pre_data, batch_size=batch_size, shuffle=False)

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        num = 0
        pre_data = np.zeros(shape=(size, len(sensor_vars)))
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                x, y = data
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                pre_data[num * batch_size:(num + 1) * batch_size, :] = pred.cpu().data.numpy()
                num += 1
        test_loss /= num_batches
        test_loss_list.append(test_loss)
        print(f'Test loss: {test_loss}')

        xs_res = xs.values - pre_data
        df_data_scaled[sensor_vars] = xs_res

        # df_data_scaled = df_denoise(df_data_scaled, sensor_vars, method='SG')

        if res_square:
            df_data_scaled[sensor_vars] = df_data_scaled[sensor_vars].applymap(lambda x_res: x_res ** 2)

        if scaled is not None:
            df_data_scaled[sensor_vars] = df_data_scaled[sensor_vars].applymap(lambda x_res: x_res * scaled)

    if normal_res is True:
        df_data_scaled = data_standardization(df_data_scaled, data_vars=sensor_vars, mode='StandardScaler')

    return df_data_scaled, df_correlation, base_structure


def graph_structure(raw_adjacency, data_var, fig_path: list = None, picture_format='png'):
    g = nx.Graph(raw_adjacency, node_size=15, font_size=8)
    g.add_nodes_from(data_var)
    # 注: pyvis 的一些操作会改变 g , 所以请放在后面, 且pyvis版本最好为0.2.1
    graph_data = utils.from_networkx(g)
    pos = nx.shell_layout(g)
    # weights = nx.get_edge_attributes(g, 'weight')
    if fig_path is not None:
        nx.draw(g, pos, node_color='pink', with_labels=True, edge_color='cornflowerblue', alpha=0.8,
                width=[float(d['weight'] * 5) for (u, v, d) in g.edges(data=True)])
        plt.savefig(fig_path[0], format=picture_format, dpi=300)
        nt = Network()
        nt.from_nx(g)
        nt.show(fig_path[1])
        plt.close()

    return graph_data


def data_standardization(df_data, data_vars, pre_scaler=None, filepath=None, mode='MinMaxScaler'):
    """
    将数据标准差归一化，去除数据异常值，降低残差数据集噪音，让数据分布在均值为0，标准差为1，聚焦于数据的差异化
    :param df_data: 残差数据
    :param data_vars: 需要标准化的数据名
    :param pre_scaler: 之前存储的 scaler 文件
    :param filepath: 保存 scaler 文件路径
    :param mode: 规范化模式 (标准化 StandardScaler or 归一化 MinMaxScaler)
    :return:
    """
    print("-------分隔符--------")
    df_scaled = df_data.copy()

    mask = (df_data.flag == 1)
    if mode == 'MinMaxScaler':
        print('数据规范化：归一化处理')
        if pre_scaler is None:
            data_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1))
            train_scaled = data_scaled.fit_transform(df_data.loc[mask, data_vars])
        else:
            data_scaled = joblib.load(pre_scaler)
            train_scaled = data_scaled.transform(df_data.loc[mask, data_vars])
    elif mode == 'StandardScaler':
        print('数据规范化：标准化处理')
        if pre_scaler is None:
            data_scaled = preprocessing.StandardScaler()
            train_scaled = data_scaled.fit_transform(df_data.loc[mask, data_vars])
        else:
            data_scaled = joblib.load(pre_scaler)
            train_scaled = data_scaled.transform(df_data.loc[mask, data_vars])
    else:
        print('Please input the correct mode!')
        return exit()

    df_scaled.loc[mask, data_vars] = train_scaled

    mask = (df_data.flag == 0)
    test_scaled = data_scaled.transform(df_data.loc[mask, data_vars])
    df_scaled.loc[mask, data_vars] = test_scaled

    if filepath is not None:
        joblib.dump(data_scaled, filepath)

    print('数据规范化已完成！')
    print("-------分隔符--------")

    return df_scaled


def split_train_test(df_data, data_var, label_var='RUL', hs_flag=False):
    if hs_flag is False:
        train_dataset = df_data[data_var].loc[df_data.flag == 1]
        y_train_label = df_data[label_var].loc[df_data.flag == 1]
        test_dataset = df_data[data_var].loc[df_data.flag == 0]
        y_test_label = df_data[label_var].loc[df_data.flag == 0]
    else:
        train_dataset = df_data[data_var].loc[(df_data.flag == 1) & (df_data.hs == 1)]
        y_train_label = df_data[label_var].loc[(df_data.flag == 1) & (df_data.hs == 1)]
        test_dataset = df_data[data_var].loc[(df_data.flag == 0) & (df_data.hs == 1)]
        y_test_label = df_data[label_var].loc[(df_data.flag == 0) & (df_data.hs == 1)]

    return train_dataset, test_dataset, y_train_label, y_test_label


class Similarity:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.dim = len(data.columns)
        self.matrix = [[0] * self.dim for _ in range(self.dim)]

    def cosine_similarity(self) -> pd.DataFrame:
        cols = self.data.columns
        idx = cols.copy()
        mat = self.data.to_numpy(dtype=float, na_value=np.nan, copy=False)
        for i in range(self.dim):
            for j in range(i, self.dim):
                vec1 = mat[:, i]
                vec2 = mat[:, j]
                self.matrix[i][j] = self.matrix[j][i] = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return pd.DataFrame(data=self.matrix, index=idx, columns=cols)

    def gaussian_kernel_similarity(self, sigma=1.0) -> pd.DataFrame:
        cols = self.data.columns
        idx = cols.copy()
        mat = self.data.to_numpy(dtype=float, na_value=np.nan, copy=False)
        for i in range(self.dim):
            for j in range(i, self.dim):
                vec1 = mat[:, i]
                vec2 = mat[:, j]
                distance = np.linalg.norm(vec1 - vec2)
                self.matrix[i][j] = self.matrix[j][i] = np.exp(-distance**2 / (2 * (sigma**2)))
        return pd.DataFrame(data=self.matrix, index=idx, columns=cols)


def corr_analysis(df_data, corr_data_filename, threshold=0.5, method='spearman'):
    """
    This function is used to generate the correlation coefficients between the variables.
    此函数用于生成变量间的相关系数
    :param df_data:待探究关系的数据，数据格式为 Dataframe。全局来看此处输入的为训练集数据（4个操作变量 + 14个可观测传感器变量）
    :param corr_data_filename:保存和读取的文件路径，如果文件已存在则直接读取 csv 文件，否则计算并保存 csv 文件
    :param threshold: 相关性判定阈值，大于等于该值视为相关，反之则不相关
    :param method: 计算相似度量的方法
    :return:返回各变量之间的关系矩阵（将每一个传感器变量看作一个节点，该矩阵就是传感器的图邻接矩阵）
    """

    if os.path.exists(corr_data_filename) is True:
        print(f'The file called {corr_data_filename} already exists!')
        df_corr_data = pd.read_csv(corr_data_filename, header=0, index_col=0)
    else:
        if method == 'spearman':
            df_corr_data = df_data.corr('spearman')
            df_corr_data.to_csv(corr_data_filename, index=True)
        elif method == 'cosine':
            s_class = Similarity(df_data)
            df_corr_data = s_class.cosine_similarity()
            df_corr_data.to_csv(corr_data_filename, index=True)
        elif method == 'gaussian kernel':
            s_class = Similarity(df_data)
            df_corr_data = s_class.gaussian_kernel_similarity(sigma=500)
            df_corr_data.to_csv(corr_data_filename, index=True)
        else:
            df_corr_data = None
            exit('请输入正确的相似性计算方法: 1.spearman; 2.cosine; 3.gaussian kernel')
        print(f'The file called {corr_data_filename} has been successfully saved!')

    print(df_corr_data.head(5))
    print('Successful correlation analysis! ')

    # 定义相关性小于 0.5 为无关，大于 0.5 的为相关
    for i in np.arange(0, len(df_corr_data)):
        for j in np.arange(0, len(df_corr_data)):
            if i == j:
                df_corr_data.iat[i, j] = 0
            elif threshold >= df_corr_data.iat[i, j] >= -threshold:
                df_corr_data.iat[i, j] = 0

    # df_corr_data.to_csv('02.csv', index=True)

    if 'time' in df_corr_data.columns:
        df_corr_data['time'] = 0
        df_corr_data.loc['time', :] = 0

    print(df_corr_data.head())

    return df_corr_data


def generate_graph_data(df_data, flag, window, data_var, graph):
    data_flag = None
    if (flag == 'train' or 1) is True:
        flag = 1
        data_flag = 'train'
    elif (flag == 'test' or 0) is True:
        flag = 0
        data_flag = 'test'
    else:
        ValueError("Invalid parameter input, please input the correct flag parameter: 'train' or 1; 'test' or 0.")

    data_list = []
    units = pd.unique(df_data[df_data.flag == flag].unit)
    print(f"The {data_flag} data units: ", units)
    for unit in units:
        cycles = pd.unique(df_data[df_data.unit == unit].cycle)
        y = pd.unique(df_data[df_data.unit == unit].RUL)
        # print("cycles: ", cycles)
        # print("RUL: ", y)
        for cycle in cycles:
            data_cycle = df_data.loc[(df_data.unit == unit) & (df_data.cycle == cycle)]
            length = len(data_cycle)
            print(f"The unit {unit}, cycle {cycle}: the length of the data is ", length)

            start_index = data_cycle.index[0]
            end_index = start_index + length
            slide_step = 10
            start_index = start_index + window
            for i in range(start_index, end_index, slide_step):
                index = range(i - window, i)
                dataset_temp = data_cycle[data_var].loc[index, :]
                node_x = torch.Tensor(np.array(dataset_temp).transpose())
                graph_y = torch.Tensor(np.unique(data_cycle.RUL))
                data_temp = Data(x=node_x, edge_index=graph.edge_index, edge_weight=graph.weight, y=graph_y)
                data_list.append(data_temp)

    return data_list


def unit_loader(df_data, data_var, graph, t_var=None, flag='train', window=32, slide_step=10, pre_cycle=1,
                device='cpu'):
    data_flag = None
    if (flag == 'train' or 1) is True:
        flag = 1
        data_flag = 'train'
    elif (flag == 'test' or 0) is True:
        flag = 0
        data_flag = 'test'
    else:
        ValueError("Invalid parameter input, please input the correct flag parameter: 'train' or 1; 'test' or 0.")
    data_list = []
    data_cycle_list = []
    data_unit_list = []
    data_subset_list = []
    data_list_temp = []
    sets = pd.unique(df_data.set)
    for subset in sets:
        print(f'The SubDataset is {subset}')
        df_sub_data = df_data[df_data.set == subset]
        units = pd.unique(df_sub_data[df_sub_data.flag == flag].unit)
        print(f"The {data_flag}ing data units of this SubDataset is: ", units)

        for unit in units:
            cycles = pd.unique(df_sub_data[df_sub_data.unit == unit].cycle)
            # 找到该单元下开始退化的时间
            df_unit_hs_cycle = df_sub_data[(df_sub_data.unit == unit) & (df_sub_data.hs == 1)].cycle
            ts = df_unit_hs_cycle.max()

            for cycle in cycles:
                data_dict = {}
                data_cycle = df_sub_data.loc[(df_sub_data.unit == unit) & (df_sub_data.cycle == cycle)]
                length = len(data_cycle)
                print(f"{subset}: The unit {unit}, cycle {cycle}: the length of the data is ", length)

                start_index = data_cycle.index[0]
                end_index = start_index + length + 1
                slide_step = slide_step
                start_index = start_index + window

                for i in range(start_index, end_index, slide_step):
                    index = range(i - window, i)
                    dataset_temp = data_cycle[data_var].loc[index, :]
                    node_x = torch.Tensor(np.array(dataset_temp).transpose()).to(device)
                    rul_label = torch.Tensor(np.unique(data_cycle.RUL))
                    t_temp = np.unique(data_cycle[t_var].values, axis=0)
                    t_tensor = torch.Tensor(t_temp).squeeze()
                    graph_y = torch.cat((rul_label, t_tensor), 0)
                    graph_y = graph_y.to(device)
                    data_temp = Data(x=node_x, edge_index=graph.edge_index, edge_weight=graph.weight, y=graph_y)
                    data_list_temp.append(data_temp)

                batch = len(data_list_temp)
                data_cycle_graph = DataLoader(data_list_temp, batch_size=batch)
                data_list_temp = []

                data_dict['set'] = subset
                data_dict['unit'] = unit
                data_dict['cycle'] = cycle
                data_dict['ts'] = ts
                data_dict['signal'] = data_cycle_graph
                data_dict['RUL'] = np.unique(data_cycle.RUL).squeeze()
                data_cycle_list.append(data_dict)

            data_unit_list.append(data_cycle_list)
            data_cycle_list = []

        data_subset_list.append(data_unit_list)
        data_unit_list = []

    return data_subset_list


def data_for_train_test(data, x_cycle, y_cycle=None, w=None):

    data_lstm_list = []
    data_unit_list = []
    data_subset_list = []
    if w is not None:
        for data_sub in data:
            for data_unit in data_sub:
                data_len = len(data_unit)
                start = 0
                slide = 1
                end = start + data_len - x_cycle + 1
                start = start + x_cycle
                for i in range(start, end, slide):
                    x = []
                    y = []
                    k = 0
                    lstm_list = data_unit[i - x_cycle:i]
                    for one_dict in lstm_list:
                        x_temp = one_dict['signal']
                        x.append(x_temp)
                    for j in range(i - x_cycle, i - x_cycle + y_cycle):
                        cycle_dict = data_unit[j]
                        y_temp = {'RUL': cycle_dict['RUL'],
                                  'cycle': cycle_dict['cycle'],
                                  'unit': cycle_dict['unit'],
                                  'set': cycle_dict['set'],
                                  'ts': cycle_dict['ts'],
                                  'w': w[k]}
                        y.append(y_temp)
                        k += 1

                    data_lstm_list.append((x, y))
                data_unit_list.append(data_lstm_list)
                data_lstm_list = []
            data_subset_list.append(data_unit_list)
            data_unit_list = []
    else:
        for data_sub in data:
            for data_unit in data_sub:
                data_len = len(data_unit)
                start = 0
                slide = 1
                end = start + data_len + 1
                start = start + x_cycle
                for i in range(start, end, slide):
                    x = []
                    y = []
                    k = 0
                    lstm_list = data_unit[i - x_cycle:i]
                    for one_dict in lstm_list:
                        x_temp = one_dict['signal']
                        x.append(x_temp)
                    for j in range(i - x_cycle, i):
                        cycle_dict = data_unit[j]
                        y_temp = {'RUL': cycle_dict['RUL'],
                                  'cycle': cycle_dict['cycle'],
                                  'unit': cycle_dict['unit'],
                                  'set': cycle_dict['set'],
                                  'ts': cycle_dict['ts'],
                                  'w': int(1)}
                        y.append(y_temp)
                        k += 1
                    data_lstm_list.append((x, y))
                data_unit_list.append(data_lstm_list)
                data_lstm_list = []
            data_subset_list.append(data_unit_list)
            data_unit_list = []

    return data_subset_list


def rmse(pred, true):
    return np.sqrt(np.mean((pred-true)**2))


def mse(pred, true):
    mse_sum = np.sum((pred - true) ** 2)
    mse_average = np.sum((pred - true) ** 2) / len(pred)
    return mse_sum, mse_average


def nasafn(pred, true):
    sum_in = 0
    for i in range(len(pred)):
        if pred[i] < true[i]:
            sum_in += np.exp((1/13)*(np.abs(pred[i]-true[i])))
        else:
            sum_in += np.exp((1/10)*(np.abs(pred[i]-true[i])))
    nasafn_average = sum_in/(len(pred)) - 1
    return sum_in, nasafn_average


def phm_score(rmse_value, nasafn_value):
    score = 0.5 * rmse_value + 0.5 * nasafn_value
    return score


# DTW
def fn_dtw(pred, true):
    template = true
    query = pred
    ds = dtw(query, template, keep_internals=True)
    cost = ds.distance
    return cost


def to_dense(data):
    """
    Returns a copy of the data object in Dense form.
    :param data:
    :return:
    """
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data


def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def wavelet_noising(data_np):
    """
    小波变换去噪，提升预测的准确性
    :param data_np: 原始数据，格式是 DataFrame
    :return:
    """
    w = pywt.Wavelet('dB10')    # 选择dB10小波基
    usecoeffs = []
    data = data_np.tolist()  # 将np.ndarray()转为列表
    ca3, cd3, cd2, cd1 = pywt.wavedec(data, w, level=3)  # 3层小波分解
    length1 = len(cd1)
    length0 = len(data)

    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs.append(ca3)

    # 软阈值方法
    for k in range(length1):
        # cd1[k] = 0.0
        if abs(cd1[k]) >= lamda / np.log2(2):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda / np.log2(2))
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if abs(cd2[k]) >= lamda / np.log2(3):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda / np.log2(3))
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        # cd3[k] = 0.0
        if abs(cd3[k]) >= lamda / np.log2(4):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda / np.log2(4))
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构

    if len(recoeffs) == len(data_np):
        recoeffs = recoeffs
    elif len(recoeffs) > len(data_np):
        recoeffs = recoeffs[0:len(data_np)]
    else:
        len_res = len(data_np) - len(recoeffs)
        recoeffs = np.append(recoeffs, recoeffs[-1].repeat(len_res, 0))
    return recoeffs


def wavelet_noising2(data_np):
    """
    小波变换去噪，提升预测的准确性
    :param data_np: 原始数据，格式是 DataFrame
    :return:
    """
    w = pywt.Wavelet('dB10')    # 选择dB10小波基
    threshold = 0.1
    max_level = pywt.dwt_max_level(len(data_np), w.dec_len)
    coeffs = pywt.wavedec(data_np, w, level=max_level)  # 小波分解
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

    recoeffs = pywt.waverec(coeffs, w)  # 信号重构

    if len(recoeffs) == len(data_np):
        recoeffs = recoeffs
    elif len(recoeffs) > len(data_np):
        recoeffs = recoeffs[0:len(data_np)]
    else:
        len_res = len(data_np) - len(recoeffs)
        recoeffs = np.append(recoeffs, recoeffs[-1].repeat(len_res, 0))
    return recoeffs


def fft_denoise(data):
    n = len(data)
    sample_rate = 10
    yf = np.fft.rfft(data)
    freq = np.fft.rfftfreq(n, d=1./sample_rate)
    # plt.plot(freq, np.abs(yf))
    # plt.show()
    yf_abs = np.abs(yf)
    indices = yf_abs > 200  # filter out those value under 100
    yf_clean = indices * yf  # noise frequency will be set to 0
    # plt.plot(freq, np.abs(yf_clean))
    # plt.show()
    denosise_data = np.fft.irfft(yf_clean)
    if len(denosise_data) == len(data):
        denosise_data = denosise_data
    elif len(denosise_data) > len(data):
        denosise_data = denosise_data[0:len(data)]
    else:
        len_res = len(data) - len(denosise_data)
        denosise_data = np.append(denosise_data, denosise_data[-1].repeat(len_res, 0))

    # plt.plot(data)
    # plt.plot(denosise_data)
    # plt.show()
    return denosise_data


def data_iter(df_data):

    index_list = []
    dataset_list = pd.unique(df_data.set)
    for subset in dataset_list:
        units = pd.unique(df_data[df_data.set == subset].unit)
        for unit in units:
            df_unit = df_data.loc[(df_data.set == subset) & (df_data.unit == unit)]
            cycles = pd.unique(df_unit.cycle)
            for cycle in cycles:
                index_tuple = (subset, unit, cycle)
                index_list.append(index_tuple)

    return index_list


def df_denoise(df_data, data_var, method='wavelet', window_length=50, unit_mode=True, adaptive_window=False):
    """
    数据集执行滤波去噪，提升预测的准确性
    :param df_data: 原始数据，格式是 DataFrame
    :param data_var: 需要进行去噪的数据列名，格式是 list
    :param method: 滤波方法选择，默认小波变换去噪 ('wavelet'), 还有：Savitzky-Golay平滑滤波 ('SG'), 傅里叶滤波 ('fft')
    :param window_length: SG平滑滤波的时间窗
    :param unit_mode: 选择按单元滤波
    :param adaptive_window: 自适应滑窗，根据输入的单元序列自动调整
    :return:
    """
    df_denoise_data = df_data.copy()

    a_list = data_iter(df_data)
    # 这样写很慢
    for name in data_var:
        denoise_loop = tqdm.tqdm(a_list, desc='denoise processing')
        for (subset, unit, cycle) in denoise_loop:
            denoise_loop.set_description(f"Denoise processing: sensor name: {name}")
            denoise_loop.set_postfix(Subset=subset, unit=unit, cycle=cycle)
            # print(f'processing: subset: {subset}, unit:{unit}, cycle:{cycle}')
            mask = ((df_data.set == subset) & (df_data.unit == unit) & (df_data.cycle == cycle))
            df_cycle = df_data.loc[mask]
            raw_data = df_cycle[name].values
            if method == 'wavelet':
                # 小波变换去噪
                denoise_data = wavelet_noising(raw_data)
            elif method == 'SG':
                # Savitzky-Golay平滑滤波
                # denoise_data = scipy.signal.savgol_filter(raw_data, window_length=window_length, polyorder=2)
                if adaptive_window:
                    window_length = len(raw_data)
                    denoise_data = scipy.signal.savgol_filter(raw_data, window_length=window_length,
                                                              polyorder=2)
                else:
                    denoise_data = scipy.signal.savgol_filter(raw_data, window_length=window_length,
                                                              polyorder=2)
            elif method == 'fft':
                # fft 傅里叶滤波
                denoise_data = fft_denoise(raw_data)
            else:
                print('Please enter the correct method!')
                return exit()

            mask = ((df_data.set == subset) & (df_data.unit == unit) & (df_data.cycle == cycle))
            df_denoise_data.loc[mask, name] = denoise_data

    print('The denoising operation has been completed!')
    return df_denoise_data


def denoise_single_sensor(df_data, name: str, group, method='wavelet',
                          window_length=50, adaptive_window=False):
    df_denoise_data = df_data.copy()
    # denoise_loop = tqdm.tqdm(group, desc='denoise processing')
    for (subset, unit, cycle) in group:
        # denoise_loop.set_description(f"Denoise processing: sensor name: {name}")
        # denoise_loop.set_postfix(Subset=subset, unit=unit, cycle=cycle)
        mask = ((df_data.set == subset) & (df_data.unit == unit) & (df_data.cycle == cycle))
        df_cycle = df_data.loc[mask]
        raw_data = df_cycle[name].values
        if method == 'wavelet':
            # 小波变换去噪
            denoise_data = wavelet_noising(raw_data)
        elif method == 'SG':
            if adaptive_window:
                window_length = len(raw_data)
                denoise_data = scipy.signal.savgol_filter(raw_data, window_length=window_length,
                                                          polyorder=2)
            else:
                denoise_data = scipy.signal.savgol_filter(raw_data, window_length=window_length,
                                                          polyorder=2)
        elif method == 'fft':
            # fft 傅里叶滤波
            denoise_data = fft_denoise(raw_data)
        else:
            print('Please enter the correct method!')
            return exit()
        mask = ((df_data.set == subset) & (df_data.unit == unit) & (df_data.cycle == cycle))
        df_denoise_data.loc[mask, name] = denoise_data
    return df_denoise_data


class MyThread(threading.Thread):
    def __init__(self, func, args=(), kwargs=None):
        super(MyThread, self).__init__()
        if kwargs is None:
            kwargs = {}
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        self.result = self.func(*self.args, **self.kwargs)

    def get_result(self):
        return self.result


def df_denoise_multithread(df_data, data_var, method='wavelet', window_length=50, unit_mode=True, adaptive_window=False):
    """
    数据集执行滤波去噪，提升预测的准确性
    :param df_data: 原始数据，格式是 DataFrame
    :param data_var: 需要进行去噪的数据列名，格式是 list
    :param method: 滤波方法选择，默认小波变换去噪 ('wavelet'), 还有：Savitzky-Golay平滑滤波 ('SG'), 傅里叶滤波 ('fft')
    :param window_length: SG平滑滤波的时间窗
    :param unit_mode: 选择按单元滤波
    :param adaptive_window: 自适应滑窗，根据输入的单元序列自动调整
    :return:
    """
    df_denoise_data = df_data.copy()
    a_list = data_iter(df_data)
    thread_list = []
    for name in data_var:
        df_sensor = df_data[['set', 'unit', 'cycle', name]]
        thread_temp = MyThread(denoise_single_sensor, kwargs={
            'df_data': df_sensor,
            'name': name,
            'group': a_list,
            'method': method,
            'window_length': window_length,
            'adaptive_window': adaptive_window
        })
        thread_list.append(thread_temp)

    for thread_temp in thread_list:
        thread_temp.start()

    for thread_temp in thread_list:
        thread_temp.join()

    for thread_temp in thread_list:
        df_denoise_senor_data = thread_temp.get_result()
        denoise_senor_data_var = df_denoise_senor_data.columns[-1]
        print(f'sensor name: {denoise_senor_data_var}')
        df_denoise_data[denoise_senor_data_var] = np.array(df_denoise_senor_data[denoise_senor_data_var])

    return df_denoise_data


def plot_df_single_color(data, variables, labels, size=12, labelsize=17, name=None, picture_format='png'):
    """
    """
    # plt.clf()
    input_dim = len(variables)
    cols = min(np.floor(input_dim ** 0.5).astype(int), 4)
    rows = (np.ceil(input_dim / cols)).astype(int)
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(size, max(size, rows * 2)))

    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        ax.plot(data[variables[n]], marker='.', markerfacecolor='none', alpha=0.7)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)
        plt.ylabel(labels[n], fontsize=labelsize)
        plt.xlabel('Time [s]', fontsize=labelsize)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, format=picture_format, dpi=300)
    # plt.show()
    plt.close()


def plot_df_color_per_unit(data, variables, labels, size=7, labelsize=17, option='Time', name=None):
    """
    """
    plt.clf()
    input_dim = len(variables)
    cols = min(np.floor(input_dim ** 0.5).astype(int), 4)
    rows = (np.ceil(input_dim / cols)).astype(int)
    gs = gridspec.GridSpec(rows, cols)
    leg = []
    fig = plt.figure(figsize=(size, max(size, rows * 2)))
    color_dic_unit = {'Unit 1': 'C0', 'Unit 2': 'C1', 'Unit 3': 'C2', 'Unit 4': 'C3', 'Unit 5': 'C4', 'Unit 6': 'C5',
                      'Unit 7': 'C6', 'Unit 8': 'C7', 'Unit 9': 'C8', 'Unit 10': 'C9', 'Unit 11': 'C10',
                      'Unit 12': 'C11', 'Unit 13': 'C12', 'Unit 14': 'C13', 'Unit 15': 'C14', 'Unit 16': 'C15',
                      'Unit 17': 'C16', 'Unit 18': 'C17', 'Unit 19': 'C18', 'Unit 20': 'C19'}

    unit_sel = np.unique(data['unit'])
    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        for j in unit_sel:
            data_unit = data.loc[data['unit'] == j]
            if option == 'cycle':
                time_s = data.loc[data['unit'] == j, 'cycle']
                label_x = 'Time [cycle]'
            else:
                time_s = np.arange(len(data_unit))
                label_x = 'Time [s]'
            ax.plot(time_s, data_unit[variables[n]], '-o', color=color_dic_unit['Unit ' + str(int(j))],
                    alpha=0.7, markersize=5)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)
            leg.append('Unit ' + str(int(j)))
        plt.ylabel(labels[n], fontsize=labelsize)
        plt.xlabel(label_x, fontsize=labelsize)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if n == 0:
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend(leg, loc='best', fontsize=labelsize - 2)  # lower left
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, format='png', dpi=300)
    plt.show()
    plt.close()


def plt_rul(df_rul_data, data_path, fig_path, picture_format='png', scale=1):

    df_rul_data_copy = df_rul_data.copy()
    df_y = pd.DataFrame(columns=['unit', 'cycle', 'RUL_true', 'RUL_prediction'])
    df_rul_data_copy['rul_1'] = df_rul_data['RUL_true'] * df_rul_data['w'] * scale
    df_rul_data_copy['rul_2'] = df_rul_data['RUL_prediction'] * df_rul_data['w'] * scale
    units = pd.unique(df_rul_data['unit'])
    for unit in units:
        cycles = pd.unique(df_rul_data[df_rul_data.unit == unit].cycle)
        for cycle in cycles:
            mask = (df_rul_data.unit == unit) & (df_rul_data.cycle == cycle)
            y_true_temp = df_rul_data_copy.loc[mask, 'rul_1'].sum(axis=0) / df_rul_data_copy.loc[mask, 'w'].sum(axis=0)
            y_pre_temp = df_rul_data_copy.loc[mask, 'rul_2'].sum(axis=0) / df_rul_data_copy.loc[mask, 'w'].sum(axis=0)

            raw = np.array([unit, cycle, y_true_temp, y_pre_temp])
            df_raw = pd.DataFrame(raw).T
            df_raw.columns = ['unit', 'cycle', 'RUL_true', 'RUL_prediction']

            df_y = pd.concat([df_y, df_raw], axis=0)

    df_y.reset_index(drop=True, inplace=True)

    # Save predictions and ground truths for further use
    df_y.to_excel(data_path)

    rmse_v = rmse(df_y['RUL_prediction'], df_y['RUL_true'])
    print("RMSE : {}".format(rmse_v))
    mse_sum, mse_average = mse(df_y['RUL_prediction'], df_y['RUL_true'])
    print("MSE : sum {}, average {}".format(mse_sum, mse_average))
    nasa_sum, nasa_average = nasafn(df_y['RUL_prediction'], df_y['RUL_true'])
    print("NASAsfn : sum {}, average {}".format(nasa_sum, nasa_average))
    nasa_score = phm_score(rmse_v, nasa_average)
    print("score : {}".format(nasa_score))
    dtw_cost = fn_dtw(df_y['RUL_prediction'], df_y['RUL_true'])
    print("DTW : {}".format(dtw_cost))

    plt.plot(df_y['RUL_true'], label='True RUL')
    plt.plot(df_y['RUL_prediction'], label='Prediction RUL')
    plt.title('RUL')
    plt.legend()
    plt.savefig(fig_path, format=picture_format, dpi=300)
    plt.close()


def plt_rul_all_data(df_rul_data, data_path, fig_path, picture_format='png', scale=1):

    df_rul_data_copy = df_rul_data.copy()
    df_y = pd.DataFrame(columns=['set', 'unit', 'cycle', 'RUL_true', 'RUL_prediction'])
    df_rul_data_copy['rul_1'] = df_rul_data['RUL_true'] * df_rul_data['w'] * scale
    df_rul_data_copy['rul_2'] = df_rul_data['RUL_prediction'] * df_rul_data['w'] * scale
    sets = pd.unique(df_rul_data['set'])
    for subset in sets:
        units = pd.unique(df_rul_data[df_rul_data.set == subset].unit)
        for unit in units:
            cycles = pd.unique(df_rul_data[(df_rul_data.set == subset) & (df_rul_data.unit == unit)].cycle)
            for cycle in cycles:
                mask = (df_rul_data.set == subset) & (df_rul_data.unit == unit) & (df_rul_data.cycle == cycle)
                y_true_temp = df_rul_data_copy.loc[mask, 'rul_1'].sum(axis=0) / df_rul_data_copy.loc[mask, 'w'].sum(axis=0)
                y_pre_temp = df_rul_data_copy.loc[mask, 'rul_2'].sum(axis=0) / df_rul_data_copy.loc[mask, 'w'].sum(axis=0)

                raw = np.array([subset, unit, cycle, y_true_temp, y_pre_temp])
                df_raw = pd.DataFrame(raw).T
                df_raw.columns = ['set', 'unit', 'cycle', 'RUL_true', 'RUL_prediction']
                df_raw[['unit', 'cycle', 'RUL_true', 'RUL_prediction']] = \
                    df_raw[['unit', 'cycle', 'RUL_true', 'RUL_prediction']].astype('float32')

                df_y = pd.concat([df_y, df_raw], axis=0)

    df_y.reset_index(drop=True, inplace=True)

    # Save predictions and ground truths for further use
    df_y.to_excel(data_path)

    rul_true = np.array(df_y['RUL_true'])
    rul_pred = np.array(df_y['RUL_prediction'])

    rmse_v = rmse(rul_pred, rul_true)
    print("RMSE : {}".format(rmse_v))
    mse_sum, mse_average = mse(rul_pred, rul_true)
    print("MSE : sum {}, average {}".format(mse_sum, mse_average))
    nasa_sum, nasa_average = nasafn(rul_pred, rul_true)
    print("NASAsfn : sum {}, average {}".format(nasa_sum, nasa_average))
    nasa_score = phm_score(rmse_v, nasa_average)
    print("score : {}".format(nasa_score))
    dtw_cost = fn_dtw(rul_pred, rul_true)
    print("DTW : {}".format(dtw_cost))

    plt.plot(df_y['RUL_true'], label='True RUL')
    plt.plot(df_y['RUL_prediction'], label='Prediction RUL')
    plt.title('RUL')
    plt.legend()
    plt.savefig(fig_path, format=picture_format, dpi=300)
    plt.close()


def plt_loss(loss_data, data_path, fig_path, picture_format='png'):
    np.savetxt(data_path, loss_data, delimiter=',')
    plt.plot(loss_data, label='train loss')
    plt.title('Train loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(fig_path, format=picture_format, dpi=300)
    plt.close()


def plt_lr(lr, data_path, fig_path, picture_format='png'):
    np.savetxt(data_path, lr, delimiter=',')
    # 画出lr的变化
    plt.plot(lr, label='learning rate')
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate's curve changes as epoch goes on!")
    plt.savefig(fig_path, format=picture_format, dpi=300)
    plt.close()


def save_df_attn(attn_list, data_path_raw=None, data_path_mean=None, data_var=None):

    df_attn = pd.DataFrame(columns=['set', 'unit', 'cycle', 'subgraph weight'] + data_var)
    df_attn_mean = pd.DataFrame(columns=['set', 'unit', 'cycle'] + data_var)
    for attn_unit_list in attn_list:
        for attn_cycle_dict in attn_unit_list:
            attn_w = attn_cycle_dict['attention weight'].reshape(-1, len(data_var))
            subset = np.array(attn_cycle_dict['set']).reshape(1, 1).repeat(attn_w.shape[0], 0)
            unit = attn_cycle_dict['unit'].reshape(1, 1).repeat(attn_w.shape[0], 0)
            cycle = attn_cycle_dict['cycle'].reshape(1, 1).repeat(attn_w.shape[0], 0)
            subgraph_weight = attn_cycle_dict['subgraph weight'].reshape(1, 1).repeat(attn_w.shape[0], 0)
            raw = np.concatenate((subset, unit, cycle, subgraph_weight, attn_w), axis=1)
            df_raw = pd.DataFrame(data=raw, columns=df_attn.columns)
            df_raw[['unit', 'cycle', 'subgraph weight'] + data_var] = \
                df_raw[['unit', 'cycle', 'subgraph weight'] + data_var].astype('float32')
            df_attn = pd.concat([df_attn, df_raw], axis=0)

    df_attn.reset_index(drop=True, inplace=True)
    df_attn.to_csv(data_path_raw)

    subsets = pd.unique(df_attn.set)
    for subset in subsets:
        units = pd.unique(df_attn[df_attn.set == subset].unit)
        sub_temp = np.array(subset).reshape(1, 1)
        for unit in units:
            cycles = pd.unique(df_attn[(df_attn.set == subset) & (df_attn.unit == unit)].cycle)
            for cycle in cycles:
                mask = (df_attn.set == subset) & (df_attn.unit == unit) & (df_attn.cycle == cycle)
                attn_temp1 = np.array(df_attn.loc[mask, data_var])
                attn_temp2 = np.array(df_attn.loc[mask, 'subgraph weight']).reshape(-1, 1)
                attn_temp = attn_temp1 * attn_temp2.repeat(attn_temp1.shape[-1], axis=1)
                attn_sum = np.sum(attn_temp, axis=0)
                attn_mean = attn_sum / np.sum(attn_temp2, axis=0).reshape(-1, 1).repeat(attn_sum.shape[-1], axis=1)

                mean = np.concatenate((sub_temp, unit.reshape(1, 1), cycle.reshape(1, 1),
                                       attn_mean.reshape(1, len(data_var))), axis=1)
                df_mean = pd.DataFrame(data=mean, columns=df_attn_mean.columns)
                df_mean[['unit', 'cycle'] + data_var] = df_mean[['unit', 'cycle'] + data_var].astype('float32')
                df_attn_mean = pd.concat([df_attn_mean, df_mean], axis=0)

    df_attn_mean.reset_index(drop=True, inplace=True)
    df_attn_mean.to_csv(data_path_mean)


class ComparedDataset(Dataset):
    """N-CMAPSS dataset for compared methods"""
    def __init__(self, data_dict_list):
        super(ComparedDataset, self).__init__()
        self.data = data_dict_list

    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, item):
        data = self.data[item]
        return data


def data_preprocess2(raw_data, data_var, device="cpu", pre_model=None,
                     scaler_file_list: list = None, res_square=False,
                     scaled: int = None, normal_res: bool = False, picture_format='png'):
    """
    数据预处理, 用于对比实验的数据预测处理
    :param raw_data: 原始数据
    :param data_var: 需要变成图数据的变量
    :param device: 计算设备
    :param pre_model: 预处理的模型文件，根据条件变量计算对应的传感器变量值并求残差特征
    :param scaler_file_list: 预处理归一化文件
    :param res_square: 残差平方处理，默认为：False
    :param scaled: 预处理后的残差特征放大倍数，默认为：None
    :param normal_res: 将残差特征标准化处理，默认为：False
    :param picture_format: 图片保存的格式
    :return:
        df_data_scaled：可以用于图数据化的数据集
    """

    operation_vars = ['alt', 'Mach', 'TRA', 'T2']
    sensor_vars = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    t_vars = ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
              'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
              'LPT_eff_mod', 'LPT_flow_mod']

    # 将数据进行分割，分为训练集和测试集
    df_train_data, df_test_data, train_label, test_label = split_train_test(raw_data, data_var)
    print(df_train_data.keys())

    # 分别对运行工况数据和传感器监测数据进行归一化处理
    df_data_scaled = data_standardization(raw_data, data_vars=operation_vars,
                                          mode='MinMaxScaler', pre_scaler=scaler_file_list[0])
    df_data_scaled = data_standardization(df_data_scaled, data_vars=sensor_vars,
                                          mode='MinMaxScaler', pre_scaler=scaler_file_list[1])

    if pre_model is not None:
        batch_size = 128
        model = PreModel(len(operation_vars), len(sensor_vars)).to(device)
        model.load_state_dict(torch.load(pre_model))

        test_loss_list = []
        loss_fn = torch.nn.MSELoss()
        w = df_data_scaled[operation_vars]
        xs = df_data_scaled[sensor_vars]

        pre_data = PreMyDataset(np.array(w), np.array(xs), device=device)
        dataloader = DataLoader(pre_data, batch_size=batch_size, shuffle=False)

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        num = 0
        pre_data = np.zeros(shape=(size, len(sensor_vars)))
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                x, y = data
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                pre_data[num * batch_size:(num + 1) * batch_size, :] = pred.cpu().data.numpy()
                num += 1
        test_loss /= num_batches
        test_loss_list.append(test_loss)
        print(f'Test loss: {test_loss}')

        xs_res = xs.values - pre_data
        df_data_scaled[sensor_vars] = xs_res

        # df_data_scaled = df_denoise(df_data_scaled, sensor_vars, method='SG')

        if res_square:
            df_data_scaled[sensor_vars] = df_data_scaled[sensor_vars].applymap(lambda x_res: x_res ** 2)

        if scaled is not None:
            df_data_scaled[sensor_vars] = df_data_scaled[sensor_vars].applymap(lambda x_res: x_res * scaled)

    if normal_res is True:
        df_data_scaled = data_standardization(df_data_scaled, data_vars=sensor_vars, mode='StandardScaler')

    return df_data_scaled


def unit_loader_for_compared_method(df_data, data_var, flag='train',
                                    window=32, slide_step=10, device='cpu'):
    data_flag = None
    if (flag == 'train' or 1) is True:
        flag = 1
        data_flag = 'train'
    elif (flag == 'test' or 0) is True:
        flag = 0
        data_flag = 'test'
    else:
        ValueError("Invalid parameter input, please input the correct flag parameter: 'train' or 1; 'test' or 0.")

    data_dict_list = []
    sets = pd.unique(df_data.set)
    for subset in sets:
        print(f'The SubDataset is {subset}')
        df_sub_data = df_data[df_data.set == subset]
        units = pd.unique(df_sub_data[df_sub_data.flag == flag].unit)
        print(f"The {data_flag}ing data units of this SubDataset is: ", units)

        for unit in units:
            cycles = pd.unique(df_sub_data[df_sub_data.unit == unit].cycle)
            # 找到该单元下开始退化的时间
            df_unit_hs_cycle = df_sub_data[(df_sub_data.unit == unit) & (df_sub_data.hs == 1)].cycle
            ts = df_unit_hs_cycle.max()

            for cycle in cycles:
                data_dict = {}
                data_cycle = df_sub_data.loc[(df_sub_data.unit == unit) & (df_sub_data.cycle == cycle)]
                length = len(data_cycle)
                print(f"{subset}: The unit {unit}, cycle {cycle}: the length of the data is ", length)

                start_index = data_cycle.index[0]
                end_index = start_index + length + 1
                slide_step = slide_step
                start_index = start_index + window

                for i in range(start_index, end_index, slide_step):
                    index = range(i - window, i)
                    dataset_temp = data_cycle[data_var].loc[index, :]
                    sensor_data = torch.Tensor(np.array(dataset_temp)).to(device)
                    rul_data = torch.Tensor(np.unique(data_cycle.RUL))

                    data_dict['set'] = subset
                    data_dict['unit'] = unit
                    data_dict['cycle'] = cycle
                    data_dict['ts'] = ts
                    data_dict['signal'] = sensor_data
                    data_dict['RUL'] = rul_data
                    data_dict_list.append(data_dict)

    return data_dict_list
