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
    def __init__(self, input_dim=400, hidden_dim=64, n_class=2):
        super(vanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1)
        self.linear = nn.Linear(hidden_dim,n_class)

    def forward(self, x, x_len):
        batchsize, timesteps, features = x.shape
        packed = pack_padded_sequence(x, x_len, batch_first=True)
        # print(packed)
        packed_output, (h_n, h_c) = self.rnn(packed)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        r_out2 = self.linear(lstm_out.view(-1, self.hidden_dim))

        return F.log_softmax(r_out2, dim=1)

class BiLSTM(nn.Module):
    def __init__(self, input_dim=400, lstm_layer=2, hidden_dim_1=256,
                 dropout_rate=0.5, hidden_dim_2=64, n_class=2):
        super(BiLSTM, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1 // 2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
            num_layers=lstm_layer)
        self.linear = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_dim_2, n_class)

    def forward(self, x, x_len):
        x = self.dropout_layer(x)
        packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        hidden_out = self.linear(lstm_out.view(-1, self.hidden_dim_1))
        dropout = self.dropout_layer(F.relu(hidden_out))
        class_out = self.output(dropout)

        return F.log_softmax(class_out, dim=1)

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
    def __init__(self, input_dim=400, num_heads=4,
                 dropout_rate=0.3, n_class=2):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout_rate)
        self.output = nn.Linear(input_dim, n_class)

    def forward(self, x, x_len):
        # x = self.dropout_layer(x)
        x = x.transpose(0,1)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0,1)
        x = self.output(x.reshape(-1, self.input_dim))
        return F.log_softmax(x, dim=1)

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
