# -*- coding: utf-8 -*-
# author: www.pinakinathc.me

import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == "__main__":
    from spectral_norm import SpectralNorm
else:
    from models.spectral_norm import SpectralNorm

class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W) # shape: B*N*out_features
        N = h.size()[1]

        f_1 = torch.matmul(h, self.a1).expand(h.shape[0], N, N) # shape: B*N*N
        f_2 = torch.matmul(h, self.a2).expand(h.shape[0], N, N) # shape: B*N*N
        e = self.leakyrelu(f_1 + f_2.transpose(1,2)) # shape: B*N*N

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GATModel(nn.Module):
    def __init__(self, feature_dim, nodes, use_gpu=True, hidden_dim=512, n_heads=3, alpha=0.2, dropout=0.6):
        """
            cuda, hidden_dim, n_heads, alpha, dropout can be made default
            feature_dim:  represents the number of channels

            nodes: [X, Y] where X represents number of graph nodes along H
                            Y represents number of graph nodes along W
                            
        """
        super(GATModel, self).__init__()
        self.feature_channel = feature_dim
        self.nodes_h, self.nodes_w = nodes

        self.n_nodes = self.nodes_h * self.nodes_w
        self.adj = self.get_adj(self.n_nodes)
        # self.max_pool_nodes = nn.MaxPool2d((kernel_h, kernel_w), stride=(stride_h, stride_w), return_indices=True)
        # self.upconv = nn.MaxUnpool2d((kernel_h, kernel_w), stride=(stride_h, stride_w))
        use_spect = True if self.training else False
        self.fuse_conv = nn.Conv2d(self.feature_channel*2, self.feature_channel, 3, 1, 1)
        if use_gpu:
            self.adj = self.adj.cuda()
            self.fuse_conv = self.fuse_conv.cuda()
        self.model = GAT(
            nfeat=self.feature_channel,
            nhid=hidden_dim,
            nclass=self.feature_channel,
            dropout=dropout,
            nheads=n_heads,
            alpha=alpha)

    def get_adj(self, nodes): # Create the adjacency matrix
        return torch.ones((nodes, nodes))

    def forward(self, x):
        _, _, feature_h, feature_w = x.size()
        stride_h = int(feature_h // self.nodes_h)
        kernel_h = feature_h - stride_h*(self.nodes_h-1)
        stride_w = int(feature_w // self.nodes_w)
        kernel_w = feature_w - stride_w*(self.nodes_w-1)

        nodes, indices = nn.functional.max_pool2d(x,
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                return_indices=True)
        # nodes, indices = self.max_pool_nodes(x)
        nodes = nodes.view(-1, self.feature_channel, self.n_nodes)
        nodes = nodes.permute(0, 2, 1)
        output = self.model(nodes, self.adj)
        feature_map = nodes.permute(0, 2, 1)
        feature_map = feature_map.view(-1, self.feature_channel,
                self.nodes_h, self.nodes_w)
        # feature_map = self.upconv(feature_map, indices)
        feature_map = nn.functional.max_unpool2d(feature_map,
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                indices=indices)
        use_spect = True if self.training else False
        output = self.fuse_conv(torch.cat((feature_map, x), 1)) # concat along channels
        return output

    def spectral_norm(self, module, use_spect=True):
        """use spectral normal layer to stable the training process"""
        if use_spect:
            return SpectralNorm(module)
        else:
            return module


if __name__ == "__main__":
    print ("testing modules using dummy data only...")
    use_gpu = torch.cuda.is_available()

    def __intra_testing():
        print ("testing intra domain...")
        # B, N, D = 1, 2, 3
        # x = torch.randn(B, N, D)
        N, D = 20, 256
        x = torch.randn(1, 256, 15, 20)
        if use_gpu:
            print ("gpu found and hence using")
            x = x.cuda()

        # intradomain = IntraDomain(N, D, D)
        intradomain = GATModel(D, (10, 10))

        output = intradomain(x)
        print ("output shape of x:{}".format(output.shape))

    __intra_testing()
