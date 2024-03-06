import os
import random
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
from NetworkClass import MAGF_GATv2, MAGF_GCN, MAGF_GAT
from data_preprocess import data_preprocess, unit_loader, plt_rul_all_data, data_for_train_test, \
                            plt_lr, plt_rul, plt_loss, save_df_attn, load_all_data_to_pd, \
                            plot_df_single_color, df_denoise_multithread


def loss_function(prediction, true, time_temp):
    scale = 1
    loss_mse = F.mse_loss(prediction, true)
    loss_1 = sum((prediction - true) ** 2 * time_temp)
    # loss_1 = sum((prediction - true) ** 2 * time_temp) + sum(abs(prediction - true))
    return scale * loss_1, loss_mse


if __name__ == '__main__':
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    operation_vars = ['alt', 'Mach', 'TRA', 'T2']
    sensor_vars = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    T_vars = ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
              'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
              'LPT_eff_mod', 'LPT_flow_mod']
    time_var = ['time']

    data_vars = sensor_vars

    sample = 10
    node_features = 64
    vector_features = 0
    node_num = len(data_vars)
    batch_size = 1

    method = 'MAGF-SPA'
    out_path = f'./results/{method}'
    results_path = os.path.join(out_path, "test")
    os.makedirs(results_path, exist_ok=True)

    dataset_list = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07', 'DS08a', 'DS08c']
    # dataset_list = ['DS01']

    print("-------分隔符--------")
    dataset_path = '../dataset/'
    df_data_raw = load_all_data_to_pd(dataset_path, dataset_list, sample_value=sample, optimize_label=True)
    # 基于斯皮尔曼相关性分析的邻接矩阵
    corr_filename = "./model/preprocessing/all_graph_data_NoOperation.csv"

    pre_model_filename = "./model/preprocessing/health_model/all_pre_model_state_dict_new.pth"
    pre_model_w_hs_scaler = './model/preprocessing/health_model/W_hs_scaler.pkl'
    pre_model_xs_hs_scaler = './model/preprocessing/health_model/Xs_hs_scaler.pkl'
    scaler_file_list = [pre_model_w_hs_scaler, pre_model_xs_hs_scaler]
    topological_save_png = f'./results/{method}/topological_graph_denoise_res_data.png'
    topological_save_html = f"./results/{method}/sensor_top.html"
    topological_save_path = [topological_save_png, topological_save_html]
    print("Dataset file path: ", dataset_path)
    print("Correlation file path: ", corr_filename)
    print("Preprocessing model file path: ", pre_model_filename)
    print("Preprocessing scaler w file path: ", scaler_file_list[0])
    print("Preprocessing scaler Xs file path: ", scaler_file_list[1])
    print("-------分隔符--------")

    df_data_scaled, df_corr, base_graph = data_preprocess(df_data_raw, data_vars, corr_filename, threshold=0.8,
                                                          pre_model=pre_model_filename, device=Device,
                                                          scaler_file_list=scaler_file_list, normal_res=False,
                                                          graph_path=topological_save_path)

    # df_data_scaled = df_denoise_multithread(df_data_scaled, data_vars, method='SG', adaptive_window=True)
    enhanced_data_path = './model/preprocessing/enhanced_data.feather'
    df_data_scaled = pd.read_feather(enhanced_data_path)
    # df_data_scaled = df_data_scaled[df_data_scaled.set == 'DS01']

    # for subset in dataset_list:
    #     units = np.unique(df_data_scaled[df_data_scaled.set == subset].unit)
    #     print('dataset: ', subset)
    #     print('units: ', units)
    #     for unit in units:
    #         df_xs_scaled_u = df_data_scaled.loc[(df_data_scaled.set == subset) &
    #                                             (df_data_scaled.unit == unit)]
    #         df_xs_scaled_u.reset_index(drop=True, inplace=True)
    #         plot_df_single_color(df_xs_scaled_u, sensor_vars, sensor_vars,
    #                              name=f"./results/all_data/{subset}_enhanced_standardization_data_unit-"+str(unit)+".png")

    print("-------分隔符--------")

    # 制作自己的图数据集，每个节点的特征向量为 1*node_features 的时间序列，邻接矩阵和上面一致
    slide_step = node_features
    time_window = node_features
    train_unit_loader_list = unit_loader(df_data_scaled, data_var=data_vars, t_var=T_vars, graph=base_graph,
                                         flag='train', window=time_window, slide_step=slide_step, device=Device)

    test_unit_loader_list = unit_loader(df_data_scaled, data_var=data_vars, t_var=T_vars, graph=base_graph,
                                        flag='test', window=time_window, slide_step=slide_step, device=Device)

    w_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    train_all_list = data_for_train_test(train_unit_loader_list, x_cycle=8, y_cycle=16, w=w_list)
    test_all_list = data_for_train_test(test_unit_loader_list, x_cycle=8, y_cycle=16, w=w_list)

    model = MAGF_GATv2(node_features, output=16, lstm_hidden_dim=32, temperature=0.1,
                       dropout=0.2, device=Device, head=4).to(Device)
    # model = MAGF_GAT(node_features, output=16, lstm_hidden_dim=32, temperature=0.1,
    #                  dropout=0.2, device=Device, head=4).to(Device)
    # model = MAGF_GCN(node_features, output=16, lstm_hidden_dim=32, dropout=0.2, device=Device).to(Device)
    epochs = 300
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                           verbose=False, threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    # train
    model.train()
    train_losses = []
    lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化
    train_list = []
    for train_subset_list in train_all_list:
        for train_unit_list in train_subset_list:
            for train_lstm_list in train_unit_list:
                train_list.append(train_lstm_list)

    steps = len(train_list)

    t_1 = time.process_time()

    for epoch in range(epochs):
        step = 0
        train_loss = 0
        train_mse = 0
        random.shuffle(train_list)
        loop_train = tqdm(train_list, desc='train')
        for LSTM_tuple in loop_train:
            LSTM_x_list = LSTM_tuple[0]
            LSTM_RUL = []
            LSTM_cycle = []
            LSTM_unit = []
            LSTM_ts = []
            LSTM_w = []
            for LSTM_dict in LSTM_tuple[1]:
                LSTM_RUL.append(LSTM_dict['RUL'])
                LSTM_cycle.append(LSTM_dict['cycle'])
                LSTM_unit.append(LSTM_dict['unit'])
                LSTM_ts.append(LSTM_dict['ts'])
                LSTM_w.append(LSTM_dict['w'])
            rul_true = torch.tensor(np.array(LSTM_RUL)).to(Device)
            cycler_true = torch.tensor(np.array(LSTM_cycle)).to(Device)
            unit_true = torch.tensor(np.array(LSTM_unit)).to(Device)
            w_true = torch.tensor(np.array(LSTM_w)).to(Device)
            ts = torch.tensor(np.array(LSTM_ts)).to(Device)

            y_out, _, _ = model(LSTM_x_list)
            loss, mse = loss_function(y_out, rul_true, cycler_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse += mse.item()

            step += 1
            loop_train.set_description(f"Epoch [{epoch+1}/{epochs}], step [{step}/{steps}]")
            loop_train.set_postfix(loss=train_loss/step, mse=train_mse/step)

        epoch_loss = train_loss / steps
        train_losses.append(epoch_loss)
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch_loss)

    t_2 = time.process_time()

    torch.save(model.state_dict(),
               os.path.join(out_path, f"{method}_model_parameter_all_data.pth"))  # 只保存模型的参数
    torch.save(model,
               os.path.join(out_path, f"{method}_model_all_data.pth"))  # 保存整个模型

    print("training time: ", t_2 - t_1)

    train_loss_path = os.path.join(out_path, f"{method}_model_all_data_train_losses.txt")
    train_loss_fig_path = os.path.join(out_path, f"{method}_model_all_data_train_loss.png")
    plt_loss(train_losses, train_loss_path, train_loss_fig_path)
    lr_path = os.path.join(out_path, f"{method}_model_all_data_lr.txt")
    lr_fig_path = os.path.join(out_path, f"{method}_model_all_data_lr.png")
    lr_list = np.hstack(lr_list)
    plt_lr(lr_list, lr_path, lr_fig_path)

    model.eval()
    df_rul = pd.DataFrame(columns=['set', 'unit', 'cycle', 'RUL_true', 'RUL_prediction', 'w'])
    df_temp = pd.DataFrame(columns=['set', 'unit', 'cycle', 'RUL_true', 'RUL_prediction', 'w'])
    attn_node_list = []
    attn_time_list = []
    for test_subset_list in test_all_list:
        for test_unit_list in test_subset_list:
            test_list = test_unit_list
            for LSTM_tuple in test_list:
                LSTM_x_list = LSTM_tuple[0]
                LSTM_set = []
                LSTM_RUL = []
                LSTM_cycle = []
                LSTM_unit = []
                LSTM_w = []
                for LSTM_dict in LSTM_tuple[1]:
                    subset = LSTM_dict['set']
                    LSTM_set.append(LSTM_dict['set'])
                    LSTM_RUL.append(LSTM_dict['RUL'])
                    LSTM_cycle.append(LSTM_dict['cycle'])
                    LSTM_unit.append(LSTM_dict['unit'])
                    LSTM_w.append(LSTM_dict['w'])
                rul_true = torch.tensor(np.array(LSTM_RUL)).to(Device)
                cycler_true = torch.tensor(np.array(LSTM_cycle)).to(Device)
                unit_true = torch.tensor(np.array(LSTM_unit)).to(Device)
                w_true = torch.tensor(np.array(LSTM_w)).to(Device)
                df_temp['set'] = np.array(LSTM_set)
                df_temp['unit'] = np.array(LSTM_unit)
                df_temp['cycle'] = np.array(LSTM_cycle)
                df_temp['RUL_true'] = np.array(LSTM_RUL)
                df_temp['w'] = np.array(LSTM_w)

                y_out, lstm_out, attn_node = model(LSTM_x_list, df_temp)
                attn_node_list.append(attn_node)

                rul_pre = y_out.cpu().data.numpy()
                df_temp['RUL_prediction'] = rul_pre
                df_rul = pd.concat([df_rul, df_temp], axis=0)

        df_rul.reset_index(drop=True, inplace=True)
        df_rul.to_excel(os.path.join(out_path, f"{method}_df_rul_raw_all_data.xlsx"))

        attn_w_raw_path = os.path.join(out_path, f"{method}_model_Attention_Weight_raw_all_data.csv")
        attn_w_mean_path = os.path.join(out_path, f"{method}_model_Attention_Weight_mean_all_data.csv")
        save_df_attn(attn_node_list, attn_w_raw_path, attn_w_mean_path, sensor_vars)

    subsets = pd.unique(df_rul.set)
    for subset in subsets:
        print(f'Dataset: {subset}')

        df_rul_sub = df_rul[df_rul.set == subset]

        rul_path = os.path.join(out_path, f"{method}_model_df_rul_{subset}.xlsx")
        rul_fig_path = os.path.join(out_path, f"{method}_model_{subset}.png")
        print('Normalized Results: ')
        plt_rul(df_rul_sub, rul_path, rul_fig_path)

        rul_path_2 = os.path.join(out_path, f"{method}_model_df_rul_{subset}_100.xlsx")
        rul_fig_path_2 = os.path.join(out_path, f"{method}_model_{subset}_100.png")
        print('Percentage Results: ')
        plt_rul(df_rul_sub, rul_path_2, rul_fig_path_2, scale=100)

    rul_path = os.path.join(out_path, f"{method}_model_df_rul_all_data.xlsx")
    rul_fig_path = os.path.join(out_path, f"{method}_model_all_data.png")
    print('Normalized Results: ')
    plt_rul_all_data(df_rul, rul_path, rul_fig_path)

    rul_path_2 = os.path.join(out_path, f"{method}_model_df_rul_all_data_100.xlsx")
    rul_fig_path_2 = os.path.join(out_path, f"{method}_model_all_data_100.png")
    print('Percentage Results: ')
    plt_rul_all_data(df_rul, rul_path_2, rul_fig_path_2, scale=100)

    print('ok')


