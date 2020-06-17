# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class Node_S(object):
    def __init__(self, idx=None):
        self.idx = idx
        self.parsent = []
        self.perior_time_list = []
        self.word_index = []

class STRNN(nn.Module):
    # def __init__(self, vocab_size, input_size, hidden_size, nclass):
    def __init__(self, vocab_size, input_size, hidden_size, nclass, device):
        super(STRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.embed = nn.Parameter(torch.zeros(size=(vocab_size, input_size)))
        torch.nn.init.normal(self.embed.data, std=0.1)

        # self.weight_topologic = nn.Parameter(torch.Tensor(input_size, hidden_size))
        # self.weight_temporal = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size, 1))

        # nn.init.uniform_(self.weight_topologic, -0.1, 0.1)
        # nn.init.uniform_(self.weight_temporal, -0.1, 0.1)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        # self.grucell_topologic = nn.GRUCell(input_size, hidden_size)
        self.grucell_temporal = nn.GRUCell(input_size, hidden_size)

        self.out = nn.Linear(hidden_size, nclass)
        nn.init.xavier_normal_(self.out.weight)
        print(self)

    def init_vector(self, shape):
        return nn.Parameter(torch.zeros(shape))

    def forward(self, x_index, sequences):
        sequences_len = len(x_index)
        node_embeddings = []
        for word_index in x_index:
            node_embeddings.append(torch.sum(self.embed[word_index,:], 0).float().view(1,-1)/len(word_index))

        for i in range(sequences_len):
            if i == 0:
                init_node_h = torch.unsqueeze(self.init_vector(self.hidden_size), 0)
            else:
                init_node_h = torch.cat([init_node_h, torch.unsqueeze(self.init_vector(self.hidden_size), 0)], 0)
        if self.cuda:
            init_node_h = init_node_h.cuda()
        # h_0 = self.init_vector(self.hidden_size)
        # output = []
        for node in sequences:
            current_embeding = node_embeddings[node[0]]
            parsent_list = node[1]
            perior_time_list = node[2]
            topological_h0 = self.init_vector(self.hidden_size)
            temporal_h0 = self.init_vector(self.hidden_size)
            if self.cuda:
                topological_h0 = topological_h0.cuda()
                temporal_h0 = temporal_h0.cuda()
            for idx in parsent_list:
                if idx != -1:
                    topological_h0 += init_node_h[idx]
            topological_h0 = topological_h0/len(parsent_list)
            for idx in perior_time_list:
                if idx != -1:
                    temporal_h0 += init_node_h[idx]
            temporal_h0 = temporal_h0/len(perior_time_list)

            topological_h0 = torch.unsqueeze(topological_h0, 0)
            temporal_h0 = torch.unsqueeze(temporal_h0, 0)

            h_1 = self.grucell_temporal(current_embeding, temporal_h0)

            h_cat = torch.cat([topological_h0, h_1], 0)
            u = torch.tanh(torch.matmul(h_cat, self.weight))
            att = torch.matmul(u, self.weight_proj)
            att_score = F.softmax(att, dim=0)
            scored_x = h_cat * att_score
            h_1 = torch.unsqueeze(torch.sum(scored_x, dim = 0), 0)

            init_node_h[node[0]] = h_1

            # h1_topologic = self.grucell_topologic(current_embeding, topological_h0)
            # h1_temporal = self.grucell_temporal(current_embeding, temporal_h0)
            #
            # h_cat = torch.cat([h1_topologic, h1_temporal], 0)
            # u = torch.tanh(torch.matmul(h_cat, self.weight))
            # att = torch.matmul(u, self.weight_proj)
            # att_score = F.softmax(att, dim=0)
            # scored_x = h_cat * att_score
            # h_1 = torch.unsqueeze(torch.sum(scored_x, dim = 0), 0)

            # init_node_h[node[0]] = h_1

        output = init_node_h[-1:]

        return F.log_softmax(self.out(output), dim=1)
