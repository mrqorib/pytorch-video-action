import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_dim=400, n_class=2):
        super(vanillaLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1)
        self.linear = nn.Linear(64,n_class)

    def forward(self, x):
        timesteps, features = x.size()
        r_in = x.view(timesteps, 1, features)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)
