import time
import os
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, gridspec
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from data_preprocess import split_train_test, load_all_data_to_pd, df_denoise, find_steady_state_data


class MyDataset(Dataset):
    def __init__(self, input_data, output_data, device):
        super(MyDataset, self).__init__()
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


class PreModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape, dropout=0.2):
        super(PreModel, self).__init__()
        self.fcn1 = nn.Sequential(nn.Linear(input_shape, 128), nn.ReLU(True))
        self.fcn2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(True))
        # self.dropout = nn.Dropout(p=dropout)
        self.fcn3 = nn.Linear(64, output_shape)

    def forward(self, x):
        h = self.fcn1(x)
        h = self.fcn2(h)
        # h = self.dropout(h)
        h = self.fcn3(h)

        return h


def train(net, dataloader, loss_fn, optimizer, epoch_num):
    size = len(dataloader.dataset)
    # loop_train = tqdm(dataloader, desc='train')
    net.train()
    step = 0
    nums = 0
    train_loss = 0
    steps = len(dataloader)

    loop_train = tqdm(enumerate(dataloader),
                      total=len(dataloader), desc='train')

    for batch, data in loop_train:
        nums += 1
        x, y = data

        # Compute prediction error
        pre = net(x)
        loss = loss_fn(pre, y) * 1000

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        step += 1
        loop_train.set_description(f'Epoch [{epoch_num + 1}/{epochs}], step [{step}/{steps}]')
        loop_train.set_postfix(loss=train_loss / step)

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(x)
        #     print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    epoch_loss = train_loss / nums
    train_loss_list.append(epoch_loss)
    return train_loss_list


def test(net, dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    net.eval()
    test_loss, correct = 0, 0
    num = 0
    pre_data = np.zeros(shape=(size, 14))
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            x, y = data
            pred = net(x)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            pre_data[num*batch_size:(num+1)*batch_size, :] = pred.cpu().data.numpy()
            num += 1
    test_loss /= num_batches
    # correct /= size
    test_loss_list.append(test_loss)
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f'Test loss: {test_loss}')
    return test_loss_list, pre_data


def plot_df_single_color(data, variables, labels, size=12, labelsize=17, name=None):
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
        plt.savefig(name, format='png', dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':

    all_training_time = 0
    sample = 1
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    operation_vars = ['alt', 'Mach', 'TRA', 'T2']
    sensor_vars = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    dataset_list = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07', 'DS08a', 'DS08c']
    time_var = ['time']
    health_var = ['hs']
    # data_vars = operation_vars + sensor_vars + time_var
    data_vars = operation_vars + sensor_vars
    input_nums = len(operation_vars)
    output_nums = len(sensor_vars)

    dataset_path = 'D:/research/projects/01_AeroEngine_RUL/dataset/'
    df_data = load_all_data_to_pd(dataset_path, dataset_list, sample_value=sample)

    df_data_temp = df_data

    train_w_hs, test_w_hs, train_xs_hs, test_xs_hs = split_train_test(df_data_temp, operation_vars, sensor_vars,
                                                                      hs_flag=True)
    train_w, test_w, train_xs, test_xs = split_train_test(df_data_temp, operation_vars, sensor_vars,
                                                          hs_flag=False)

    # 归一化 [0, 1]
    x_train_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_w_hs_scaled = x_train_scaled.fit_transform(train_w_hs)
    train_w_scaled = x_train_scaled.transform(train_w)
    test_w_hs_scaled = x_train_scaled.transform(test_w_hs)
    test_w_scaled = x_train_scaled.transform(test_w)

    y_train_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_xs_hs_scaled = y_train_scaled.fit_transform(train_xs_hs)
    train_xs_scaled = y_train_scaled.transform(train_xs)
    test_xs_hs_scaled = y_train_scaled.transform(test_xs_hs)
    test_xs_scaled = y_train_scaled.transform(test_xs)

    # joblib.dump(x_train_scaled, f'./model/preprocessing/health_model/W_hs_scaler.pkl')
    # joblib.dump(y_train_scaled, f'./model/preprocessing/health_model/Xs_hs_scaler.pkl')

    train_data = MyDataset(train_w_hs_scaled, train_xs_hs_scaled, device=Device)
    test_data = MyDataset(test_w_scaled, test_xs_scaled, device=Device)

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print("No data preprocessing model for all datasets, so let's train it! ")
    print("Here is the training process:")

    # 初始化模型
    pre_model_path = './model/preprocessing/health_model/all_pre_model_new.pth'
    pre_model_state_dict_path = './model/preprocessing/health_model/all_pre_model_state_dict_new.pth'

    model = PreModel(input_nums, output_nums, dropout=0.2).to(Device)
    print(model)

    if os.path.exists(pre_model_state_dict_path) is not True:
        print('The model file does not exists! \nTraining...')
        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        epochs = num_epochs
        model.train()
        train_loss_list = []

        t_1 = time.process_time()
        for t in range(epochs):
            # print(f"Epoch {t + 1}\n-------------------------------")
            train(model, train_dataloader, loss_fn, optimizer, epoch_num=t)
        print("Done!")
        t_2 = time.process_time()

        time = t_2 - t_1
        print(f"The model training time: {time}")

        fig = plt.figure(figsize=(20, 10))
        plt.plot(train_loss_list, label="train loss")
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.title('Train loss')
        plt.savefig(f"./model/preprocessing/health_model/picture/all_data_preprocessing_data_train_loss.png",
                    format='png', dpi=300)
        # plt.show()

        torch.save(model.state_dict(), pre_model_state_dict_path)
        torch.save(model, pre_model_path)  # 保存整个整体训练模型
        print('Model training completed!')
        print("Saved PyTorch Model State to model.pth")
    else:
        print('The model file already exists! \nLoading...')
        model.load_state_dict(torch.load(pre_model_state_dict_path))
        print('Model loading completed! ')

    # testing
    model.eval()
    test_loss_list = []
    loss_fn = nn.MSELoss()
    testing_loss, xs_pre = test(model, test_dataloader, loss_fn)

    df_test_xs_scaled = pd.DataFrame(data=test_xs_scaled, columns=sensor_vars)
    df_test_xs_scaled['units'] = df_data_temp[df_data_temp.flag == 0].unit.values
    df_test_xs_scaled['cycle'] = df_data_temp[df_data_temp.flag == 0].cycle.values
    df_test_xs_scaled['set'] = df_data_temp[df_data_temp.flag == 0].set.values

    xs_res = test_xs_scaled - xs_pre
    df_xs_res = pd.DataFrame(data=xs_res, columns=sensor_vars)
    df_xs_res['units'] = df_data_temp[df_data_temp.flag == 0].unit.values
    df_xs_res['cycle'] = df_data_temp[df_data_temp.flag == 0].cycle.values
    df_xs_res['set'] = df_data_temp[df_data_temp.flag == 0].set.values

    for subset in dataset_list:

        units = np.unique(df_data_temp[(df_data_temp.set == subset) & (df_data_temp.flag == 0)].unit)
        print('dataset: ', subset)
        print('test units: ', units)
        for unit in units:
            df_test_xs_scaled_u = df_test_xs_scaled.loc[(df_test_xs_scaled.set == subset) &
                                                        (df_test_xs_scaled.units == unit)]
            df_test_xs_scaled_u.reset_index(drop=True, inplace=True)
            plot_df_single_color(df_test_xs_scaled_u, sensor_vars, sensor_vars,
                                 name=f"./model/preprocessing/health_model/picture/{subset}_raw_data_unit-"
                                      + str(unit) + ".png")

            df_xs_res_u = df_xs_res.loc[(df_xs_res.set == subset) &
                                        (df_xs_res.units == unit)]
            df_xs_res_u.reset_index(drop=True, inplace=True)
            plot_df_single_color(df_xs_res_u, sensor_vars, sensor_vars,
                                 name=f"./model/preprocessing/health_model/picture/{subset}_enhanced_data_unit-"
                                      + str(unit) + ".png")

