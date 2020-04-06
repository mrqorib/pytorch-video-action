import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
import torch

class SimpleFC(nn.Module):
    def __init__(self, input_dim=400, n_class=2):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class vanillaLSTM(nn.Module):
    def __init__(self, input_dim=400, lstm_layer=1, dropout_rate=0,
                 hidden_dim_1=64, hidden_dim_2=60, n_class=2, mode='cont'):
        super(vanillaLSTM, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.mode = mode
        self.lstm_layer = lstm_layer
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            batch_first=True,
            dropout=dropout_rate,
            num_layers=lstm_layer)
        self.linear = nn.Linear(hidden_dim_1,hidden_dim_2)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_dim_2, n_class)

    def forward(self, x, x_len):
        batchsize, timesteps, features = x.shape
        packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        # print(packed)
        packed_output, (h_n, h_c) = self.rnn(packed)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        if self.mode == 'last':
            h_n = h_n.view(self.lstm_layer, 1, batchsize, self.hidden_dim_1)
            lstm_out = h_n[-1,:,:,:]
            # print(lstm_out.shape)
        hidden_out = self.linear(lstm_out.view(-1, self.hidden_dim_1))
        dropout = self.dropout_layer(F.relu(hidden_out))
        class_out = self.output(dropout)
        return F.log_softmax(class_out, dim=1)

class BiLSTM(nn.Module):
    def __init__(self, input_dim=400, lstm_layer=2, hidden_dim_1=256,
                 dropout_rate=0.5, hidden_dim_2=64, n_class=2, mode='cont'):
        super(BiLSTM, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.mode = mode
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1 // 2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
            num_layers=lstm_layer)
        self.linear = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim_1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_dim_2, n_class)

    def forward(self, x, x_len):
        x = self.dropout_layer(x)
        packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        if self.mode == 'last':
            lstm_out = lstm_out[:,-1,:]
        hidden_out = self.linear(lstm_out)
        # print(lstm_out.shape)
        if self.mode == 'avg':
            hidden_out = torch.mean(hidden_out, dim=1)
        # print(lstm_out.shape)
        hidden_out = hidden_out.contiguous().view(-1, self.hidden_dim_2)
        # lstm_out = lstm_out.view(-1, self.hidden_dim_2)
        # lstm_out = self.batch_norm(lstm_out)
        dropout = self.dropout_layer(F.relu(hidden_out))
        class_out = self.output(dropout)

        return F.log_softmax(class_out, dim=1)

class BiLSTMWithLM(nn.Module):
    def __init__(self, input_dim=400, lstm_layer=2, hidden_dim_1=256,
                 dropout_rate=0.5, hidden_dim_2=64, n_class=2, context=2):
        super(BiLSTMWithLM, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.n_class = n_class
        self.context = context
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1 // 2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
            num_layers=lstm_layer)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_dim_1)
        self.linear = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dim_2)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        # self.class_context = torch.zeros(context * n_class)
        self.register_buffer('class_context', torch.zeros(context * n_class))
        self.output = nn.Linear(context * n_class + hidden_dim_2, n_class)

    def forward(self, x, x_len):
        batchsize, max_len, input_dim = x.shape
        x = self.dropout_layer(x)
        packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = lstm_out.view(-1, self.hidden_dim_1)
        lstm_out = self.batch_norm_1(lstm_out)
        hidden_out = self.linear(lstm_out)
        hidden_out = torch.tanh(hidden_out)
        hidden_out = self.batch_norm_2(hidden_out)
        # hidden_out = self.dropout_layer(hidden_out)
        final_output = torch.zeros((batchsize * max_len, self.n_class), device=x.device)
        reset_idx = np.cumsum(x_len)
        for batch in range(len(hidden_out)):
            if batch in reset_idx:
                self.class_context = torch.zeros(self.context * self.n_class,
                                                 device=x.device)
            if not self.class_context.is_cuda:
                print('{} at the begining'.format(batch))
            last_with_context = torch.cat([self.class_context, hidden_out[batch,:]])
            class_out = self.output(last_with_context)
            # print(class_out.shape)
            # assert class_out.shape == (self.n_class)
            class_out = F.log_softmax(class_out, dim=0)
            self.class_context = torch.cat([self.class_context[self.n_class:].detach(),
                                            class_out.detach()])
            if not self.class_context.is_cuda:
                print('{} at the end'.format(batch))
            final_output[batch,:] = class_out
        return final_output

class BiGRU(nn.Module):
    def __init__(self, input_dim=400, gru_layer=4, hidden_dim_1=256,
                 dropout_rate=0.5, hidden_dim_2=64, n_class=2):
        super(BiGRU, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim_1 // 2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
            num_layers=gru_layer)
        self.linear = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_dim_1, n_class)

    def forward(self, x, x_len):
        x = self.dropout_layer(x)
        packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        class_out = self.output(lstm_out.view(-1, self.hidden_dim_1))
        # dropout = self.dropout_layer(F.relu(hidden_out))
        # class_out = self.output(dropout)
        return F.log_softmax(class_out, dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=400, num_heads=4, hidden_dim=256,
                 dropout_rate=0.3, n_class=2, mode='cont'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout_rate)
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_class)

    def forward(self, x, x_len):
        # x = self.dropout_layer(x)
        batchsize, max_len, input_dim = x.shape
        x = x.transpose(0,1)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0,1)
        if self.mode == 'last':
            x = x[:,-1,:]
        x = self.hidden1(F.relu(x))
        if self.mode == 'cont': 
            x = x.contiguous().view(-1, self.hidden_dim)
        elif self.mode == 'avg':
            x = torch.mean(x, dim=1)
        x = self.output(F.relu(x))
        return F.log_softmax(x, dim=1)

class ExpWindowAttention(nn.Module):
    def __init__(self, input_dim=400, num_heads=4, n_class=2,
                 dropout_rate=0.3, window_size=5, use_previous_output=True):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.n_class = n_class
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout_rate)
        self.output = nn.Linear(input_dim, n_class)
        self.combine_output = nn.Linear(n_class * (window_size + 1), n_class)
    
    def forward(self, x, x_len):
        
        # x = self.dropout_layer(x)
        batchsize, max_len, input_dim = x.shape
        x = F.pad(x, (0, 0, 0, self.window_size), "constant", 0)
        x = x.transpose(0,1)
        # start_frame = np.array([0] + x_len) + self.window_size
        att_feats = []
        final_output = torch.zeros((max_len, batchsize, self.n_class), device=x.device)
        for f in range(self.window_size, max_len, self.window_size):
            start_frame = f - self.window_size
            end_frame = f + self.window_size + 1
            context = x[start_frame:end_frame, :, :]
            # print(context.shape)
            feat, _ = self.attention(context, context, context)
            feat = feat[self.window_size,:,:]
            assert feat.shape == (batchsize, input_dim)
            probs = self.output(feat)
            final_output[start_frame,:,:] = probs
        final_output = final_output.transpose(0,1)
        assert final_output.shape == (batchsize, max_len, self.n_class)
        final_output = final_output.reshape(-1, self.n_class)
        # print('inside: ', final_output.shape)
        return F.log_softmax(final_output, dim=1)

# class ExpWindowAttention(nn.Module):
#     def __init__(self, input_dim=400, num_heads=4, n_class=2,
#                  dropout_rate=0.3, window_size=5, use_previous_output=True):
#         super().__init__()
#         self.input_dim = input_dim
#         self.window_size = window_size
#         self.n_class = n_class
#         self.dropout_layer = nn.Dropout(p=dropout_rate)
#         self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout_rate)
#         self.output = nn.Linear(input_dim, n_class)
#         self.combine_output = nn.Linear(n_class * (window_size + 1), n_class)
    
#     def forward(self, x, x_len):
        
#         # x = self.dropout_layer(x)
#         batchsize, max_len, input_dim = x.shape
#         x = F.pad(x, (0, 0, self.window_size, self.window_size), "constant", 0)
#         x = x.transpose(0,1)
#         # start_frame = np.array([0] + x_len) + self.window_size
#         att_feats = []
#         final_output = torch.zeros((max_len, batchsize, self.n_class), device=x.device)
#         for f in range(self.window_size, max_len):
#             start_frame = f - self.window_size
#             end_frame = f + self.window_size + 1
#             context = x[start_frame:end_frame, :, :]
#             feat, _ = self.attention(context, context, context)
#             feat = feat[self.window_size,:,:]
#             # print(feat.shape)
#             assert feat.shape == (batchsize, input_dim)
#             # feat = feat.squeeze(0)
#             probs = self.output(feat)
#             att_feats.append(probs)
#             # print(att_feats)
#             # print(len(att_feats[max(start_frame, 0):f+1]))
#             context_feat = torch.cat(att_feats[max(start_frame, 0):f+1], 1)
#             # print(context_feat.shape)
#             len_diff = self.n_class * (self.window_size + 1) - context_feat.shape[1]
#             if len_diff > 0:
#                 context_feat = F.pad(context_feat, (len_diff, 0), "constant", 0)
#             # print(context_feat.shape)
#             assert context_feat.shape == (batchsize, self.n_class * (self.window_size + 1))
#             final_output[start_frame,:,:] = self.combine_output(F.relu(context_feat))
#         final_output = final_output.transpose(0,1)
#         final_output = final_output.reshape(-1, self.n_class)
#         # print('inside: ', final_output.shape)
#         return F.log_softmax(final_output, dim=1)
        # start_idx = np.cumsum([0] + x_len)
        # for idx, feat in enumerate(att_feats):
        #     if idx in start_idx:
        #         last_class = np.zeros((1,))
        
        # x = x.view(-1, self.input_dim)
        # assert x.shape == (max_len * batchsize, self.input_dim)
        # x = self.output(x)
        # return F.log_softmax(x, dim=1)

class MultiStageModel(nn.Module):
    def __init__(self, dim=400, num_stages=4, num_layers=10, num_f_maps=64, n_class=2):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, n_class)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, n_class, n_class)) for s in range(num_stages-1)])
        self.n_class = n_class

    def forward(self, x, x_len):
        x = x.transpose(1,2)
        mask = torch.zeros(x.shape[0], self.n_class, max(x_len), dtype=torch.float).cuda()
        for i in range(x.shape[0]):
            mask[i, :, :x_len[i]] = torch.ones(self.n_class, x_len[i]).cuda()

        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        outputs = outputs.permute(0, 1, 3, 2)
        outputs = outputs.contiguous().view(outputs.shape[0], outputs.shape[1]*outputs.shape[2], outputs.shape[3])
        outputs = torch.max(outputs, 0)[0]
        return outputs

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, n_class):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, n_class, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
