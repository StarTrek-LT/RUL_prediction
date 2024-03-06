import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocess import data_preprocess2, unit_loader_for_compared_method, ComparedDataset, plot_df_single_color, \
                            plt_rul_all_data, plt_lr, plt_rul, plt_loss, df_denoise_multithread, load_all_data_to_pd


class ResNetMethod3(nn.Module):
    """ paper: 'Probabilistic Remaining Useful Life Prediction Based on Deep
        Convolutional Neural Network' """

    def __init__(self, time_num, sensor_num, dropout=0.2, output_dim=1):
        super(ResNetMethod3, self).__init__()
        # input shape: [batch_size, channels=1, time_window=64, sensor_num=14]
        self.H = time_num
        self.W = sensor_num

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 1), stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # [batch_size, 8, 64, 14]
        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(16)
        )
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
        )
        self.ReLU2 = nn.ReLU(inplace=True)

        # [batch_size, 16, 32, 14]
        self.cnn_layers_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(16)
        )

        self.downsample_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
        )
        self.ReLU3 = nn.ReLU(inplace=True)

        # [batch_size, 16, 16, 14]
        self.cnn_layers_4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(32)
        )
        self.downsample_4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
        )
        self.ReLU4 = nn.ReLU(inplace=True)

        # [batch_size, 32, 16, 14]
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Sequential(
            nn.Linear(in_features=int(self.H / 4) * self.W * 32, out_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.linear_2 = nn.Linear(in_features=100, out_features=output_dim)

    def forward(self, input_data):

        h = self.cnn_layers_1(input_data)

        h_res = self.downsample_2(h)
        h = self.cnn_layers_2(h)
        h += h_res
        h = self.ReLU2(h)

        h_res = self.downsample_3(h)
        h = self.cnn_layers_3(h)
        h += h_res
        h = self.ReLU3(h)

        h_res = self.downsample_4(h)
        h = self.cnn_layers_4(h)
        h += h_res
        h = self.ReLU4(h)

        h = self.flatten(h)
        h = self.linear_1(h)
        pred = self.linear_2(h)

        return pred


if __name__ == '__main__':
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    operation_vars = ['alt', 'Mach', 'TRA', 'T2']
    sensor_vars = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    T_vars = ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
              'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
              'LPT_eff_mod', 'LPT_flow_mod']
    time_var = ['time']

    data_vars = sensor_vars
    sensor_nums = len(data_vars)

    sample = 10
    node_features = 64
    vector_features = 0
    node_num = len(data_vars)

    method = 'ResNet'
    out_path = f'./results/{method}'
    results_path = os.path.join(out_path, "test")
    os.makedirs(results_path, exist_ok=True)

    dataset_list = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07', 'DS08a', 'DS08c']
    # dataset_list = ['DS01', 'DS02']
    # dataset_list = ['DS08c']

    print("-------分隔符--------")
    dataset_path = '../dataset/'
    df_data_raw = load_all_data_to_pd(dataset_path, dataset_list, sample_value=sample, optimize_label=True)
    pre_model_filename = "./model/preprocessing/health_model/all_pre_model_state_dict_new.pth"
    pre_model_w_hs_scaler = './model/preprocessing/health_model/W_hs_scaler.pkl'
    pre_model_xs_hs_scaler = './model/preprocessing/health_model/Xs_hs_scaler.pkl'
    scaler_file_list = [pre_model_w_hs_scaler, pre_model_xs_hs_scaler]
    print("Dataset file path: ", dataset_path)
    print("Preprocessing model file path: ", pre_model_filename)
    print("Preprocessing scaler w file path: ", scaler_file_list[0])
    print("Preprocessing scaler Xs file path: ", scaler_file_list[1])
    print("-------分隔符--------")

    df_data_scaled = data_preprocess2(df_data_raw, data_vars,
                                      pre_model=pre_model_filename, device=Device,
                                      scaler_file_list=scaler_file_list, normal_res=False,
                                      )
    df_data_scaled = df_denoise_multithread(df_data_scaled, data_vars, method='SG', adaptive_window=True)

    # 制作数据集
    train_unit_loader_list = unit_loader_for_compared_method(df_data_scaled, data_var=data_vars,
                                                             flag='train', window=node_features,
                                                             slide_step=node_features, device=Device)

    test_unit_loader_list = unit_loader_for_compared_method(df_data_scaled, data_var=data_vars,
                                                            flag='test', window=node_features,
                                                            slide_step=node_features, device=Device)

    train_input_data = ComparedDataset(train_unit_loader_list)
    test_input_data = ComparedDataset(test_unit_loader_list)

    batch_size = 512

    train_dataloaders = DataLoader(train_input_data, batch_size=batch_size, num_workers=0, shuffle=True)
    test_dataloaders = DataLoader(test_input_data, batch_size=1, num_workers=0, shuffle=False)

    model = ResNetMethod3(time_num=node_features, sensor_num=sensor_nums,
                          dropout=0.2, output_dim=1).to(Device)
    epochs = 300
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # train
    model.train()
    t_1 = time.process_time()
    steps = len(train_dataloaders)
    train_losses = []
    lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化

    for epoch in range(epochs):
        step = 0
        train_loss = 0
        train_mse = 0
        loop_train = tqdm(train_dataloaders,
                          total=len(train_dataloaders), desc='train')
        for data_batch in loop_train:
            sensor_data = data_batch['signal'].to(Device)
            rul_data = data_batch['RUL'].to(Device)
            cycler_true = data_batch['cycle'].to(Device)

            sensor_data = sensor_data.unsqueeze(1)
            output_rul = model(sensor_data)

            loss = loss_fn(output_rul, rul_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1
            loop_train.set_description(f"Epoch [{epoch + 1}/{epochs}], step [{step}/{steps}]")
            loop_train.set_postfix(loss=train_loss / step)

        epoch_loss = train_loss / steps
        train_losses.append(epoch_loss)
        lr_list.append(optimizer.param_groups[0]['lr'])

    t_2 = time.process_time()
    torch.save(model.state_dict(),
               "./results/compared/ResNet_model_parameter_all_data.pth")  # 只保存模型的参数
    torch.save(model, './results/compared/ResNet_model_all_data.pth')  # 保存整个模型

    print("training time: ", t_2 - t_1)

    train_loss_path = os.path.join(out_path, f"{method}_model_all_data_train_losses.txt")
    train_loss_fig_path = os.path.join(out_path, f"{method}_model_all_data_train_loss.png")
    plt_loss(train_losses, train_loss_path, train_loss_fig_path)
    lr_path = os.path.join(out_path, f"{method}_model_all_data_lr.txt")
    lr_fig_path = os.path.join(out_path, f"{method}_model_all_data_lr.png")
    lr_list = np.hstack(lr_list)
    plt_lr(lr_list, lr_path, lr_fig_path)

    # 测试
    model.eval()
    loop_test = tqdm(test_dataloaders,
                     total=len(test_dataloaders), desc='test')
    df_rul = pd.DataFrame(columns=['set', 'unit', 'cycle', 'RUL_true', 'RUL_prediction', 'w'])
    df_temp = pd.DataFrame(columns=['set', 'unit', 'cycle', 'RUL_true', 'RUL_prediction', 'w'])

    for data_batch in loop_test:

        sensor_data = data_batch['signal'].to(Device)
        rul_data = data_batch['RUL'].to(Device)
        df_temp['set'] = np.array(data_batch['set'])
        df_temp['unit'] = np.array(data_batch['unit'])
        df_temp['cycle'] = np.array(data_batch['cycle'])
        df_temp['RUL_true'] = data_batch['RUL'].cpu().data.numpy()
        df_temp['w'] = np.array([1.0]).astype('float32')

        sensor_data = sensor_data.unsqueeze(1)
        output_rul = model(sensor_data)
        rul_pre = output_rul.cpu().data.numpy()
        df_temp['RUL_prediction'] = rul_pre

        df_rul = pd.concat([df_rul, df_temp], axis=0)

    df_rul.reset_index(drop=True, inplace=True)
    df_rul.to_excel(os.path.join(out_path, f"{method}_df_rul_raw_all_data.xlsx"))

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
