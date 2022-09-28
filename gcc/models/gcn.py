#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 15:38
# TODO:

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.model_zoo.chem.gnn import GCNLayer
from dgl.nn.pytorch import AvgPooling, Set2Set, GraphConv


class GCNLayer(nn.Module):
    """Single layer GCN for updating node features
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    activation : activation function
        Default to be ReLU
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """

    def __init__(self, in_feats, out_feats, activation=F.relu,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=False, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, feats):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv(bg, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


'''
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, order=1, act=None,
                 dropout=0, batch_norm=False, aggr="concat"):
        super(GCNLayer, self).__init__()
        self.lins = nn.ModuleList()
        self.bias = nn.ParameterList()
        for _ in range(order + 1):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=False))
            self.bias.append(nn.Parameter(torch.zeros(out_dim)))

        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.batch_norm = batch_norm
        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(order + 1):
                self.offset.append(nn.Parameter(torch.zeros(out_dim)))
                self.scale.append(nn.Parameter(torch.ones(out_dim)))

        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight)

    def feat_trans(self, features, idx):
        h = self.lins[idx](features) + self.bias[idx]

        if self.act is not None:
            h = self.act(h)

        if self.batch_norm:
            mean = h.mean(dim=1).view(h.shape[0], 1)
            var = h.var(dim=1, unbiased=False).view(h.shape[0], 1) + 1e-9
            h = (h - mean) * self.scale[idx] * torch.rsqrt(var) + self.offset[idx]

        return h

    def forward(self, graph, features):
        g = graph.local_var()
        h_in = self.dropout(features)
        h_hop = [h_in]

        D_norm = g.ndata['train_D_norm'] if 'train_D_norm' in g.ndata else g.ndata['full_D_norm']
        for _ in range(self.order):
            g.ndata['h'] = h_hop[-1]
            if 'w' not in g.edata:
                g.edata['w'] = torch.ones((g.num_edges(),)).to(features.device)
            g.update_all(fn.u_mul_e('h', 'w', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h * D_norm
            h_hop.append(h)

        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        if self.aggr == "mean":
            h_out = h_part[0]
            for i in range(len(h_part) - 1):
                h_out = h_out + h_part[i + 1]
        elif self.aggr == "concat":
            h_out = torch.cat(h_part, 1)
        else:
            raise NotImplementedError

        return h_out
'''


class UnsupervisedGCN(nn.Module):
    def __init__(
            self,
            hidden_size=64,
            num_layer=2,
            readout="avg",
            layernorm: bool = False,
            set2set_lstm_layer: int = 3,
            set2set_iter: int = 6,
    ):
        super(UnsupervisedGCN, self).__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    in_feats=hidden_size,
                    out_feats=hidden_size,
                    activation=F.relu if i + 1 < num_layer else None,
                    residual=False,
                    batchnorm=False,
                    dropout=0.0,
                )
                for i in range(num_layer)
            ]
        )
        if readout == "avg":
            self.readout = AvgPooling()
        elif readout == "set2set":
            self.readout = Set2Set(
                hidden_size, n_iters=set2set_iter, n_layers=set2set_lstm_layer
            )
            self.linear = nn.Linear(2 * hidden_size, hidden_size)
        elif readout == "root":
            # HACK: process outside the model part
            self.readout = lambda _, x: x
        else:
            raise NotImplementedError
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
            # self.ln = nn.BatchNorm1d(hidden_size, affine=False)

    def forward(self, g, feats, efeats=None):
        for layer in self.layers:
            feats = layer(g, feats)
        feats = self.readout(g, feats)
        if isinstance(self.readout, Set2Set):
            feats = self.linear(feats)
        if self.layernorm:
            feats = self.ln(feats)
        return feats


if __name__ == "__main__":
    model = UnsupervisedGCN()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    print(model(g, feat).shape)
