import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
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
