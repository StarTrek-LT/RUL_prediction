import torch
import math
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, Dropout
import torch.nn.functional as F
import torch_geometric
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn import GCNConv, GraphConv, global_add_pool, SAGPooling, SAGEConv, GATConv, global_mean_pool
import torch_geometric.nn as pyg_nn
from typing import Optional


class PreModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PreModel, self).__init__()
        self.fcn1 = nn.Sequential(Linear(input_shape, 128), nn.ReLU(True))
        self.fcn2 = nn.Sequential(Linear(128, 64), nn.ReLU(True))
        self.dropout = Dropout(0.2)
        self.fcn3 = Linear(64, output_shape)

    def forward(self, x):
        h = self.fcn1(x)
        h = self.fcn2(h)
        h = self.dropout(h)
        h = self.fcn3(h)

        return h


class MAGF_GATv2(torch.nn.Module):
    """
    MAGF:
    GATv2_layer2_Res_NodeAttentionPooling_SubgraphAttentionPooling+LSTM
    """
    def __init__(self, input_features, output, lstm_hidden_dim, device='cpu', dropout=0.0, head=1, temperature=1):
        super(MAGF_GATv2, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        # self.conv1 = GATConv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.conv1 = pyg_nn.GATv2Conv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()

        # self.conv2 = GATConv(128, 256, dropout=dropout, heads=head, concat=False)
        self.conv2 = pyg_nn.GATv2Conv(128, 256, dropout=dropout, heads=head, concat=False)
        self.LayerNormal2 = nn.LayerNorm(256)

        self.fcn_res = Linear(input_features, 256)
        self.tanh1 = nn.Tanh()

        self.AttentionalAggregation = GraphSelfAttentionalAggregation(node_nums=14, heads=head,
                                                                      temperature=temperature,
                                                                      average_attn_weights=True,
                                                                      abs_mode=False)
        self.BatchAttentionalPooling = GraphSelfAttentionalAggregation(node_nums=1, heads=head,
                                                                       temperature=temperature,
                                                                       batch_pooling_mode=True,
                                                                       average_attn_weights=True)

        self.LSTM = torch.nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=dropout, bidirectional=True)
        self.hidden = self.init_lstm_hidden(device)
        self.tanh2 = nn.Tanh()

        self.fcn1 = Linear(2 * 8 * lstm_hidden_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fcn2 = Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fcn3 = Linear(32, output)
        self.tanh3 = nn.Tanh()
        self.relu4 = nn.ReLU()

    def init_lstm_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, data, df_data=None):
        count = 0
        batch_weight = None
        attn_w_list = []
        attn_w_dict = {}
        attn_time_dict = {}
        for i, data_loader in enumerate(data):
            for dataset in data_loader:
                dataset = dataset.cuda()
                x, edge_index, edge_weight = dataset.x, dataset.edge_index, dataset.edge_weight

                batch = dataset.batch

                h = self.conv1(x, edge_index)
                h = self.LayerNormal1(h)
                h = self.relu1(h)
                h = self.conv2(h, edge_index)
                h = self.LayerNormal2(h)

                # skip connect
                h += self.fcn_res(x)
                h = self.tanh1(h)

                # graph pooling
                # h = self.global_mean_pooling(h, batch)
                h, weight_matrix1, node_weight = self.AttentionalAggregation(h, batch=batch,
                                                                             return_attention_weights=True)
                # batch_pooling
                h, weight_matrix2, batch_weight = self.BatchAttentionalPooling(h, batch=batch,
                                                                               return_attention_weights=True)

                if self.training is not True:
                    for j in range(node_weight.shape[0]):
                        attn_w_dict = dict()
                        attn_w_dict.update({
                            'set': df_data['set'].iloc[i],
                            'unit': df_data['unit'].iloc[i],
                            'cycle': df_data['cycle'].iloc[i],
                            'attention weight': node_weight.detach().cpu().data.numpy()[j]
                        })
                        if batch_weight is not None:
                            attn_w_dict['subgraph weight'] = batch_weight.detach().cpu().data.numpy()[j]
                        else:
                            attn_w_dict['subgraph weight'] = 1.0
                        attn_w_list.append(attn_w_dict)

                gcn_out = h.view(1, 1, -1)
                # gcn_out = torch.mean(h, dim=0).view(1, 1, -1)

                if count == 0:
                    lstm_input = gcn_out
                else:
                    lstm_input = torch.cat((lstm_input, gcn_out), 0)
                count += 1

        lstm_out, _ = self.LSTM(lstm_input)
        h = lstm_out

        h = self.fcn1(h.view(1, -1))
        h = self.dropout1(h)
        h = self.fcn2(h)
        h = self.dropout2(h)
        h = self.fcn3(h)

        h = self.tanh3(h)
        h = self.relu4(h)

        out = h.squeeze()

        return out, lstm_out, attn_w_list


class GATv2Layer1(torch.nn.Module):
    """
    GATv2_layer2_Res_NodeAttentionPooling_SubgraphAttentionPooling+LSTM
    """
    def __init__(self, input_features, output, lstm_hidden_dim, device='cpu', dropout=0.0, head=1, temperature=1):
        super(GATv2Layer1, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        self.conv2 = pyg_nn.GATv2Conv(input_features, 256, dropout=dropout, heads=head, concat=False)
        self.LayerNormal2 = nn.LayerNorm(256)

        self.fcn_res = Linear(input_features, 256)
        self.tanh1 = nn.Tanh()

        self.AttentionalAggregation = GraphSelfAttentionalAggregation(node_nums=14, heads=head,
                                                                      temperature=temperature,
                                                                      average_attn_weights=True,
                                                                      abs_mode=False)
        self.BatchAttentionalPooling = GraphSelfAttentionalAggregation(node_nums=1, heads=head,
                                                                       temperature=temperature,
                                                                       batch_pooling_mode=True,
                                                                       average_attn_weights=True)

        self.LSTM = torch.nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=dropout, bidirectional=True)
        self.hidden = self.init_lstm_hidden(device)
        self.tanh2 = nn.Tanh()

        self.fcn1 = Linear(2 * 8 * lstm_hidden_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fcn2 = Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fcn3 = Linear(32, output)
        self.tanh3 = nn.Tanh()
        self.relu4 = nn.ReLU()

    def init_lstm_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, data, df_data=None):
        count = 0
        batch_weight = None
        attn_w_list = []
        attn_w_dict = {}
        attn_time_dict = {}
        for i, data_loader in enumerate(data):
            for dataset in data_loader:
                dataset = dataset.cuda()
                x, edge_index, edge_weight = dataset.x, dataset.edge_index, dataset.edge_weight

                batch = dataset.batch

                h = self.conv2(x, edge_index)
                h = self.LayerNormal2(h)

                # skip connect
                h += self.fcn_res(x)
                h = self.tanh1(h)

                # graph pooling
                # h = self.global_mean_pooling(h, batch)
                h, weight_matrix1, node_weight = self.AttentionalAggregation(h, batch=batch,
                                                                             return_attention_weights=True)
                # batch_pooling
                h, weight_matrix2, batch_weight = self.BatchAttentionalPooling(h, batch=batch,
                                                                               return_attention_weights=True)

                if self.training is not True:
                    for j in range(node_weight.shape[0]):
                        attn_w_dict = dict()
                        attn_w_dict.update({
                            'set': df_data['set'].iloc[i],
                            'unit': df_data['unit'].iloc[i],
                            'cycle': df_data['cycle'].iloc[i],
                            'attention weight': node_weight.detach().cpu().data.numpy()[j]
                        })
                        if batch_weight is not None:
                            attn_w_dict['subgraph weight'] = batch_weight.detach().cpu().data.numpy()[j]
                        else:
                            attn_w_dict['subgraph weight'] = 1.0
                        attn_w_list.append(attn_w_dict)

                gcn_out = h.view(1, 1, -1)
                # gcn_out = torch.mean(h, dim=0).view(1, 1, -1)

                if count == 0:
                    lstm_input = gcn_out
                else:
                    lstm_input = torch.cat((lstm_input, gcn_out), 0)
                count += 1

        lstm_out, _ = self.LSTM(lstm_input)
        h = lstm_out

        h = self.fcn1(h.view(1, -1))
        h = self.dropout1(h)
        h = self.fcn2(h)
        h = self.dropout2(h)
        h = self.fcn3(h)

        h = self.tanh3(h)
        h = self.relu4(h)

        out = h.squeeze()

        return out, lstm_out, attn_w_list


class GATv2Layer3(torch.nn.Module):
    """
    GATv2_layer2_Res_NodeAttentionPooling_SubgraphAttentionPooling+LSTM
    """
    def __init__(self, input_features, output, lstm_hidden_dim, device='cpu', dropout=0.0, head=1, temperature=1):
        super(GATv2Layer3, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        # self.conv1 = GATConv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.conv1 = pyg_nn.GATv2Conv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()

        self.conv3 = pyg_nn.GATv2Conv(128, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal3 = nn.LayerNorm(128)
        self.relu3 = nn.ReLU()

        # self.conv2 = GATConv(128, 256, dropout=dropout, heads=head, concat=False)
        self.conv2 = pyg_nn.GATv2Conv(128, 256, dropout=dropout, heads=head, concat=False)
        self.LayerNormal2 = nn.LayerNorm(256)

        self.fcn_res = Linear(input_features, 256)
        self.tanh1 = nn.Tanh()

        self.AttentionalAggregation = GraphSelfAttentionalAggregation(node_nums=14, heads=head,
                                                                      temperature=temperature,
                                                                      average_attn_weights=True,
                                                                      abs_mode=False)
        self.BatchAttentionalPooling = GraphSelfAttentionalAggregation(node_nums=1, heads=head,
                                                                       temperature=temperature,
                                                                       batch_pooling_mode=True,
                                                                       average_attn_weights=True)

        self.LSTM = torch.nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=dropout, bidirectional=True)
        self.hidden = self.init_lstm_hidden(device)
        self.tanh2 = nn.Tanh()

        self.fcn1 = Linear(2 * 8 * lstm_hidden_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fcn2 = Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fcn3 = Linear(32, output)
        self.tanh3 = nn.Tanh()
        self.relu4 = nn.ReLU()

    def init_lstm_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, data, df_data=None):
        count = 0
        batch_weight = None
        attn_w_list = []
        attn_w_dict = {}
        attn_time_dict = {}
        for i, data_loader in enumerate(data):
            for dataset in data_loader:
                dataset = dataset.cuda()
                x, edge_index, edge_weight = dataset.x, dataset.edge_index, dataset.edge_weight

                batch = dataset.batch

                h = self.conv1(x, edge_index)
                h = self.LayerNormal1(h)
                h = self.relu1(h)
                h = self.conv3(h, edge_index)
                h = self.LayerNormal3(h)
                h = self.relu3(h)
                h = self.conv2(h, edge_index)
                h = self.LayerNormal2(h)

                # skip connect
                h += self.fcn_res(x)
                h = self.tanh1(h)

                # graph pooling
                # h = self.global_mean_pooling(h, batch)
                h, weight_matrix1, node_weight = self.AttentionalAggregation(h, batch=batch,
                                                                             return_attention_weights=True)
                # batch_pooling
                h, weight_matrix2, batch_weight = self.BatchAttentionalPooling(h, batch=batch,
                                                                               return_attention_weights=True)

                if self.training is not True:
                    for j in range(node_weight.shape[0]):
                        attn_w_dict = dict()
                        attn_w_dict.update({
                            'set': df_data['set'].iloc[i],
                            'unit': df_data['unit'].iloc[i],
                            'cycle': df_data['cycle'].iloc[i],
                            'attention weight': node_weight.detach().cpu().data.numpy()[j]
                        })
                        if batch_weight is not None:
                            attn_w_dict['subgraph weight'] = batch_weight.detach().cpu().data.numpy()[j]
                        else:
                            attn_w_dict['subgraph weight'] = 1.0
                        attn_w_list.append(attn_w_dict)

                gcn_out = h.view(1, 1, -1)
                # gcn_out = torch.mean(h, dim=0).view(1, 1, -1)

                if count == 0:
                    lstm_input = gcn_out
                else:
                    lstm_input = torch.cat((lstm_input, gcn_out), 0)
                count += 1

        lstm_out, _ = self.LSTM(lstm_input)
        h = lstm_out

        h = self.fcn1(h.view(1, -1))
        h = self.dropout1(h)
        h = self.fcn2(h)
        h = self.dropout2(h)
        h = self.fcn3(h)

        h = self.tanh3(h)
        h = self.relu4(h)

        out = h.squeeze()

        return out, lstm_out, attn_w_list


class GATv2Layer4(torch.nn.Module):
    """
    GATv2_layer2_Res_NodeAttentionPooling_SubgraphAttentionPooling+LSTM
    """
    def __init__(self, input_features, output, lstm_hidden_dim, device='cpu', dropout=0.0, head=1, temperature=1):
        super(GATv2Layer4, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        # self.conv1 = GATConv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.conv1 = pyg_nn.GATv2Conv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()

        self.conv3 = pyg_nn.GATv2Conv(128, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal3 = nn.LayerNorm(128)
        self.relu3 = nn.ReLU()

        self.conv4 = pyg_nn.GATv2Conv(128, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal4 = nn.LayerNorm(128)
        self.relu4 = nn.ReLU()

        # self.conv2 = GATConv(128, 256, dropout=dropout, heads=head, concat=False)
        self.conv2 = pyg_nn.GATv2Conv(128, 256, dropout=dropout, heads=head, concat=False)
        self.LayerNormal2 = nn.LayerNorm(256)

        self.fcn_res = Linear(input_features, 256)
        self.tanh1 = nn.Tanh()

        self.AttentionalAggregation = GraphSelfAttentionalAggregation(node_nums=14, heads=head,
                                                                      temperature=temperature,
                                                                      average_attn_weights=True,
                                                                      abs_mode=False)
        self.BatchAttentionalPooling = GraphSelfAttentionalAggregation(node_nums=1, heads=head,
                                                                       temperature=temperature,
                                                                       batch_pooling_mode=True,
                                                                       average_attn_weights=True)

        self.LSTM = torch.nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=dropout, bidirectional=True)
        self.hidden = self.init_lstm_hidden(device)
        self.tanh2 = nn.Tanh()

        self.fcn1 = Linear(2 * 8 * lstm_hidden_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fcn2 = Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fcn3 = Linear(32, output)
        self.tanh3 = nn.Tanh()
        self.relu4 = nn.ReLU()

    def init_lstm_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, data, df_data=None):
        count = 0
        batch_weight = None
        attn_w_list = []
        attn_w_dict = {}
        attn_time_dict = {}
        for i, data_loader in enumerate(data):
            for dataset in data_loader:
                dataset = dataset.cuda()
                x, edge_index, edge_weight = dataset.x, dataset.edge_index, dataset.edge_weight

                batch = dataset.batch

                h = self.conv1(x, edge_index)
                h = self.LayerNormal1(h)
                h = self.relu1(h)
                h = self.conv3(h, edge_index)
                h = self.LayerNormal3(h)
                h = self.relu3(h)
                h = self.conv4(h, edge_index)
                h = self.LayerNormal4(h)
                h = self.relu4(h)
                h = self.conv2(h, edge_index)
                h = self.LayerNormal2(h)

                # skip connect
                h += self.fcn_res(x)
                h = self.tanh1(h)

                # graph pooling
                # h = self.global_mean_pooling(h, batch)
                h, weight_matrix1, node_weight = self.AttentionalAggregation(h, batch=batch,
                                                                             return_attention_weights=True)
                # batch_pooling
                h, weight_matrix2, batch_weight = self.BatchAttentionalPooling(h, batch=batch,
                                                                               return_attention_weights=True)

                if self.training is not True:
                    for j in range(node_weight.shape[0]):
                        attn_w_dict = dict()
                        attn_w_dict.update({
                            'set': df_data['set'].iloc[i],
                            'unit': df_data['unit'].iloc[i],
                            'cycle': df_data['cycle'].iloc[i],
                            'attention weight': node_weight.detach().cpu().data.numpy()[j]
                        })
                        if batch_weight is not None:
                            attn_w_dict['subgraph weight'] = batch_weight.detach().cpu().data.numpy()[j]
                        else:
                            attn_w_dict['subgraph weight'] = 1.0
                        attn_w_list.append(attn_w_dict)

                gcn_out = h.view(1, 1, -1)
                # gcn_out = torch.mean(h, dim=0).view(1, 1, -1)

                if count == 0:
                    lstm_input = gcn_out
                else:
                    lstm_input = torch.cat((lstm_input, gcn_out), 0)
                count += 1

        lstm_out, _ = self.LSTM(lstm_input)
        h = lstm_out

        h = self.fcn1(h.view(1, -1))
        h = self.dropout1(h)
        h = self.fcn2(h)
        h = self.dropout2(h)
        h = self.fcn3(h)

        h = self.tanh3(h)
        h = self.relu4(h)

        out = h.squeeze()

        return out, lstm_out, attn_w_list


class MAGF_GAT(torch.nn.Module):
    """
    GAT_layer2_Res_NodeAttentionPooling_SubgraphAttentionPooling+LSTM
    """
    def __init__(self, input_features, output, lstm_hidden_dim, device='cpu', dropout=0.0, head=1, temperature=1):
        super(MAGF_GAT, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        self.conv1 = GATConv(input_features, 128, dropout=dropout, heads=head, concat=False)
        self.LayerNormal1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()

        self.conv2 = GATConv(128, 256, dropout=dropout, heads=head, concat=False)
        self.LayerNormal2 = nn.LayerNorm(256)

        self.fcn_res = Linear(input_features, 256)
        self.tanh1 = nn.Tanh()

        self.AttentionalAggregation = GraphSelfAttentionalAggregation(node_nums=14, heads=head,
                                                                      temperature=temperature,
                                                                      average_attn_weights=True,
                                                                      abs_mode=False)
        self.BatchAttentionalPooling = GraphSelfAttentionalAggregation(node_nums=1, heads=head,
                                                                       temperature=temperature,
                                                                       batch_pooling_mode=True,
                                                                       average_attn_weights=True)

        self.LSTM = torch.nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=dropout, bidirectional=True)
        self.hidden = self.init_lstm_hidden(device)
        self.tanh2 = nn.Tanh()

        self.fcn1 = Linear(2 * 8 * lstm_hidden_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fcn2 = Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fcn3 = Linear(32, output)
        self.tanh3 = nn.Tanh()
        self.relu4 = nn.ReLU()

    def init_lstm_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, data, df_data=None):
        count = 0
        batch_weight = None
        attn_w_list = []
        attn_w_dict = {}
        attn_time_dict = {}
        for i, data_loader in enumerate(data):
            for dataset in data_loader:
                dataset = dataset.cuda()
                x, edge_index, edge_weight = dataset.x, dataset.edge_index, dataset.edge_weight

                batch = dataset.batch

                h = self.conv1(x, edge_index)
                h = self.LayerNormal1(h)
                h = self.relu1(h)
                h = self.conv2(h, edge_index)
                h = self.LayerNormal2(h)

                # skip connect
                h += self.fcn_res(x)
                h = self.tanh1(h)

                # graph pooling
                # h = self.global_mean_pooling(h, batch)
                h, weight_matrix1, node_weight = self.AttentionalAggregation(h, batch=batch,
                                                                             return_attention_weights=True)
                # batch_pooling
                h, weight_matrix2, batch_weight = self.BatchAttentionalPooling(h, batch=batch,
                                                                               return_attention_weights=True)

                if self.training is not True:
                    for j in range(node_weight.shape[0]):
                        attn_w_dict = dict()
                        attn_w_dict.update({
                            'set': df_data['set'].iloc[i],
                            'unit': df_data['unit'].iloc[i],
                            'cycle': df_data['cycle'].iloc[i],
                            'attention weight': node_weight.detach().cpu().data.numpy()[j]
                        })
                        if batch_weight is not None:
                            attn_w_dict['subgraph weight'] = batch_weight.detach().cpu().data.numpy()[j]
                        else:
                            attn_w_dict['subgraph weight'] = 1.0
                        attn_w_list.append(attn_w_dict)

                gcn_out = h.view(1, 1, -1)
                # gcn_out = torch.mean(h, dim=0).view(1, 1, -1)

                if count == 0:
                    lstm_input = gcn_out
                else:
                    lstm_input = torch.cat((lstm_input, gcn_out), 0)
                count += 1

        lstm_out, _ = self.LSTM(lstm_input)
        h = lstm_out

        h = self.fcn1(h.view(1, -1))
        h = self.dropout1(h)
        h = self.fcn2(h)
        h = self.dropout2(h)
        h = self.fcn3(h)

        h = self.tanh3(h)
        h = self.relu4(h)

        out = h.squeeze()

        return out, lstm_out, attn_w_list


class MAGF_GCN(torch.nn.Module):
    """
    GCN_layer2_Res_NodeAttentionPooling_SubgraphAttentionPooling+LSTM
    """
    def __init__(self, input_features, output, lstm_hidden_dim, device='cpu', dropout=0.0, head=1, temperature=1):
        super(MAGF_GCN, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        self.conv1 = GCNConv(input_features, 128)
        self.LayerNormal1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()

        self.conv2 = GCNConv(128, 256)
        self.LayerNormal2 = nn.LayerNorm(256)

        self.fcn_res = Linear(input_features, 256)
        self.tanh1 = nn.Tanh()

        self.AttentionalAggregation = GraphSelfAttentionalAggregation(node_nums=14, heads=head,
                                                                      temperature=temperature,
                                                                      average_attn_weights=True,
                                                                      abs_mode=False)
        self.BatchAttentionalPooling = GraphSelfAttentionalAggregation(node_nums=1, heads=head,
                                                                       temperature=temperature,
                                                                       batch_pooling_mode=True,
                                                                       average_attn_weights=True)

        self.LSTM = torch.nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=dropout, bidirectional=True)
        self.hidden = self.init_lstm_hidden(device)
        self.tanh2 = nn.Tanh()

        self.fcn1 = Linear(2 * 8 * lstm_hidden_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fcn2 = Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fcn3 = Linear(32, output)
        self.tanh3 = nn.Tanh()
        self.relu4 = nn.ReLU()

    def init_lstm_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, data, df_data=None):
        count = 0
        batch_weight = None
        attn_w_list = []
        attn_w_dict = {}
        attn_time_dict = {}
        for i, data_loader in enumerate(data):
            for dataset in data_loader:
                dataset = dataset.cuda()
                x, edge_index, edge_weight = dataset.x, dataset.edge_index, dataset.edge_weight

                batch = dataset.batch

                h = self.conv1(x, edge_index, edge_weight)
                h = self.LayerNormal1(h)
                h = self.relu1(h)
                h = self.conv2(h, edge_index,edge_weight)
                h = self.LayerNormal2(h)

                # skip connect
                h += self.fcn_res(x)
                h = self.tanh1(h)

                # graph pooling
                # h = self.global_mean_pooling(h, batch)
                h, weight_matrix1, node_weight = self.AttentionalAggregation(h, batch=batch,
                                                                             return_attention_weights=True)
                # batch_pooling
                h, weight_matrix2, batch_weight = self.BatchAttentionalPooling(h, batch=batch,
                                                                               return_attention_weights=True)

                if self.training is not True:
                    for j in range(node_weight.shape[0]):
                        attn_w_dict = dict()
                        attn_w_dict.update({
                            'set': df_data['set'].iloc[i],
                            'unit': df_data['unit'].iloc[i],
                            'cycle': df_data['cycle'].iloc[i],
                            'attention weight': node_weight.detach().cpu().data.numpy()[j]
                        })
                        if batch_weight is not None:
                            attn_w_dict['subgraph weight'] = batch_weight.detach().cpu().data.numpy()[j]
                        else:
                            attn_w_dict['subgraph weight'] = 1.0
                        attn_w_list.append(attn_w_dict)

                gcn_out = h.view(1, 1, -1)
                # gcn_out = torch.mean(h, dim=0).view(1, 1, -1)

                if count == 0:
                    lstm_input = gcn_out
                else:
                    lstm_input = torch.cat((lstm_input, gcn_out), 0)
                count += 1

        lstm_out, _ = self.LSTM(lstm_input)
        h = lstm_out

        h = self.fcn1(h.view(1, -1))
        h = self.dropout1(h)
        h = self.fcn2(h)
        h = self.dropout2(h)
        h = self.fcn3(h)

        h = self.tanh3(h)
        h = self.relu4(h)

        out = h.squeeze()

        return out, lstm_out, attn_w_list


class GraphSelfAttentionalAggregation(torch.nn.Module):
    def __init__(
        self,
        node_nums: int = None,
        node_dim: int = None,
        temperature: int = 1,
        heads: int = 1,
        average_attn_weights: bool = False,
        batch_pooling_mode: bool = False,
        abs_mode: bool = False
    ):
        super().__init__()
        self.node_nums = node_nums
        self.node_dim = node_dim
        self.temperature = temperature
        self.heads = heads
        self.average_attn_weights = average_attn_weights
        self.batch_pooling_mode = batch_pooling_mode
        self.abs_mode = abs_mode

    def forward(self, x: Tensor, batch: OptTensor = None,
                return_attention_weights=None):

        # 输入前先将 (batch_num*node_num, embed_dim) 的格式转化为 (batch_num, node_num, embed_dim)
        if x.dim() == 2:
            assert (batch is not None) | (self.node_nums is not None), \
                "在批数量 batch 和 节点数 node_nums 两个参数必须输入其中一个!"
            if batch is not None:
                assert (self.node_nums is not None) | (self.node_dim is not None), \
                    "在批数量 batch 确定地情况下，节点数 node_nums 和节点维度 node_dim 两个参数必须输入其中一个! "
                if self.node_nums is not None:
                    batch_num = int(len(torch.unique(batch)))
                    x = x.view(batch_num, self.node_nums, -1)
                else:
                    batch_num = int(len(torch.unique(batch)))
                    x = x.view(batch_num, -1, self.node_dim)
            else:
                assert (self.node_dim is not None) & (self.node_nums is not None), \
                    "在批数量 batch 不确定地情况下，节点数 node_nums 和节点维度 node_dim 两个参数必须同时输入!"
                x = x.view(-1, self.node_nums, self.node_dim)
        else:
            assert x.dim() == 3, "输入向量的维度只能是二维 [batch*node_nums, embed_dim] 或三维 [batch, node_nums, embed_dim]!"

        if self.batch_pooling_mode:
            x = x.contiguous().transpose(0, 1)  # [node_nums, batch, embed_dim]

        # 输入部分处理 自注意力, Q K V 相同
        query = key = value = x
        batch_num, node_num, embed_dim = query.shape

        # 判断节点特征的维度是否能被注意力头数 heads 整除
        num_heads = self.heads
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        # 将 Q K V 处理成多头的形式
        q = query.contiguous().view(batch_num * num_heads, node_num, head_dim)
        k = key.contiguous().view(batch_num * num_heads, node_num, head_dim)

        q_scaled = q / math.sqrt(embed_dim)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        if self.abs_mode:
            attn_output_weights = torch.abs(attn_output_weights)
        attn_output_weights = self.softmax_t(attn_output_weights, dim=[-2, -1])
        node_attn_weights = torch.sum(attn_output_weights, dim=-1).view(q.shape[0], q.shape[1], -1)
        node_attn_weights = node_attn_weights.view(batch_num, num_heads, node_num, -1)
        attn_output_weights = attn_output_weights.view(batch_num, num_heads, node_num, -1)

        if self.average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            node_attn_weights = node_attn_weights.sum(dim=1) / num_heads

        else:
            node_attn_weights = node_attn_weights.contiguous().transpose(-3, -2).view(batch_num, node_num, -1)
            attn_output_weights = attn_output_weights.contiguous().transpose(-3, -2).view(batch_num, node_num, -1)

        v = value.contiguous().view(batch_num, node_num, embed_dim)
        out = torch.bmm(node_attn_weights.contiguous().transpose(-2, -1), v)

        if self.batch_pooling_mode:
            out = out.contiguous().transpose(0, 1)  # [1, node_nums, embed_dim]
            node_attn_weights = node_attn_weights.contiguous().transpose(0, 1)

        if return_attention_weights:
            return out, attn_output_weights, node_attn_weights
        else:
            return out

    def softmax_t(self, input_tensor: Tensor, dim: Optional[int or list] = None):
        """
        Applies a softmax function with temperature.
        Softmax is defined as:
        :math:`\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i/t)}{\\sum_j \\exp(x_j/t)}`
        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `[0, 1]` and sum to 1.
        :param input_tensor: (Tensor) input, Tensor [batch_num*num_heads, node_num, node_num]
        :param dim: Optional[int] A dimension along which softmax will be computed.
        :return: Tensor [batch_num*num_heads, node_num, node_num]
        """
        weights_t = torch.div(input_tensor, self.temperature)
        exp_weights = torch.exp(weights_t)
        sum_weights = torch.sum(exp_weights, dim).unsqueeze(-1).unsqueeze(-1).expand(weights_t.shape)
        weights_new = torch.div(exp_weights, sum_weights)

        return weights_new

