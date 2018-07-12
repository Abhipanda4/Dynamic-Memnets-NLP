import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class ScoringNetwork(nn.Module):
    def __init__(self, inp_dim, hidden_layer_size, arch=1):
        '''
        Assigns a weight to each fact based on a simple feedforward
        architecture
        '''
        super(ScoringNetwork, self).__init__()
        self.arch = arch
        # if arch == 1:
        self.fc1 = nn.Linear(4 * inp_dim, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)
        # else:
            # self.fc1 = nn.Linear(3 * inp_dim, hidden_layer_size)
            # self.fc2 = nn.Linear(hidden_layer_size, 1)

    def construct_feature_vector(self, x, M, Q):
        '''
        Given input facts, question and memory at previous timestep,
        this function computes the feature vector for feeding into the
        scoring network.
        Refer: https://arxiv.org/pdf/1603.01417.pdf (section 3.3)
        '''
        if self.arch == 1:
            M = M.expand_as(x)
            Q = Q.expand_as(x)
            new_feature = torch.cat([
                x * Q,
                x * M,
                torch.abs(x - Q),
                torch.abs(x - M)
            ], dim=2)
            return new_feature
        else:
            M = M.expand_as(x)
            Q = Q.expand_as(x)
            new_feature = torch.cat([x, M, Q], dim=2)
            return new_feature

    def forward(self, x, M, Q):
        x = self.construct_feature_vector(x, M, Q)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size()[0], -1)
        G = F.sigmoid(x)
        return G

class MemoryUpdateNetwork(nn.Module):
    def __init__(self, memory_size, hidden_layer_size):
        super(MemoryUpdateNetwork, self).__init__()
        self.fc1 = nn.Linear(3 * memory_size, hidden_layer_size)

    def forward(self, new_M, prev_M, Q):
        x = torch.cat([new_M, prev_M, Q], dim=2)
        x = F.leaky_relu(self.fc1(x))
        return x

class MemoryUpdateCell(nn.Module):
    def __init__(self, inp_dim, hidden_size):
        super(MemoryUpdateCell, self).__init__()
        self.hidden_size = hidden_size
        self.attention_cell = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, C, G):
        '''
        Iterate through each fact `C_t` till last fact is reached.
        The memory update resulting after last fact is assigned to be the
        episode tensor.

        @sizes:
        C: [batch X num_sentences X hidden_size]
        G: [batch X num_sentences]
        prev_M: [batch X 1 X hidden_size]

        M: [batch X hidden_size]
        '''
        num_sentences = C.size()[1]
        h = Variable(torch.zeros(self.hidden_size))
        for sentence in range(num_sentences):
            c_t = C[:, sentence, :]
            g_t = G[:, sentence]
            if sentence == 0:
                h = h.unsqueeze(0).expand_as(c_t)
                h = h.unsqueeze(0)
            g_t = g_t.unsqueeze(1).expand_as(h)
            c_t = c_t.unsqueeze(1)
            h = g_t * self.attention_cell(c_t, h)[1] + (1 - g_t) * h
        return h.transpose(0, 1)


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size, scoring_net_hidden_size=120, arch=1):
        super(EpisodicMemory, self).__init__()
        self.memory_update = MemoryUpdateCell(hidden_size, hidden_size)
        self.scoring_net = ScoringNetwork(hidden_size, scoring_net_hidden_size, arch)
        self.arch = arch

    def forward(self, C, Q, prev_M):
        '''
        @args:
        C: Facts received from the input module
        Q: Question vector
        prev_M: Initial memory of the module before episode

        @output:
        A tensor representing the new memory after the episode

        @sizes:
        C: [batch X num_sentences X hidden_size]
        Q: [batch X 1 X hidden_size]
        prev_M: [batch X 1 X hidden_size]

        next_M: [batch X 1 X hidden_size]
        '''
        G = self.scoring_net(C, Q, prev_M)
        next_M = self.memory_update(C, G)
        return next_M
