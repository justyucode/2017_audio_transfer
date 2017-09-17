import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

# =============================== Q(z|X) ======================================
class RecurrentEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim, r_level):
        super(RecurrentEncoder, self).__init__()
        self.lstm_layer = r_level
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.lstm = nn.LSTM(input_dim, input_dim, r_level)
        self.linear = nn.Linear(input_dim, input_dim)

        self.mean_linear = nn.Linear(input_dim, embed_dim)
        self.var_linear = nn.Linear(input_dim, embed_dim)

    def init_hidden(self, embed_dim, on_gpu=True, gpu_id=0):
        self.cell_state = Variable(torch.zeros(self.lstm_layer, 1, self.input_dim))
        self.hidden_state = Variable(torch.zeros(self.lstm_layer, 1, self.input_dim))

        if on_gpu:
            self.cell_state = self.cell_state.cuda(gpu_id)
            self.hidden_state = self.hidden_state.cuda(gpu_id)

    def reset_hidden(self):
        self.cell_state = Variable(self.cell_state.data)
        self.hidden_state = Variable(self.hidden_state.data)

    def forward(self, data):
        x = data.view(data.size(0), 1, self.input_dim)

        x, hidden = self.lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = hidden

        x = data.view(data.size(0), self.input_dim)
        x = F.elu(self.linear(x))

        mean = self.mean_linear(x)
        var = self.var_linear(x)

        return mean, var


def sample_z(mu, log_var, mb_size, Z_dim, on_gpu=True, gpu_id=0):
    eps = Variable(torch.randn(mb_size, Z_dim))
    if on_gpu:
        eps = eps.cuda(gpu_id)
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================
class RecurrentDecoder(nn.Module):

    def __init__(self, input_dim, out_embed, r_level):
        super(RecurrentDecoder, self).__init__()
        self.lstm_layer = r_level
        self.input_dim = input_dim
        self.embed_dim = out_embed

        self.lstm = nn.LSTM(input_dim, input_dim, r_level)
        self.linear = nn.Linear(input_dim, input_dim)

        self.embed_linear = nn.Linear(input_dim, out_embed)

    def init_hidden(self, embed_dim, on_gpu=True, gpu_id=0):
        self.cell_state = Variable(torch.zeros(self.lstm_layer, 1, self.input_dim))
        self.hidden_state = Variable(torch.zeros(self.lstm_layer, 1, self.input_dim))

        if on_gpu:
            self.cell_state = self.cell_state.cuda(gpu_id)
            self.hidden_state = self.hidden_state.cuda(gpu_id)

    def reset_hidden(self):
        self.cell_state = Variable(self.cell_state.data)
        self.hidden_state = Variable(self.hidden_state.data)

    def forward(self, data):
        x = data.view(data.size(0), 1, self.input_dim)

        x, hidden = self.lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = hidden

        x = data.view(data.size(0), self.input_dim)
        x = F.elu(self.linear(x))

        embed = self.embed_linear(x)

        return embed
