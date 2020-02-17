import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GUN(nn.Module):
    def __init__(
            self, num_feature, num_classes, emb_size=128, un_layer=2, if_mlp_trans=False,
            if_trans_bn=False, if_mlp_bn=True, if_trans_share=True, if_bn_share=True, if_trans_bias=False,
            if_separa=False, trans_act='leaky', mlp_act='leaky', mlp_size=128, mlp_layer=1,
            trans_init='xavier', mlp_init='xavier', bn_mom=0.1, drop_rate=0.5, device='cpu'):
        super(GUN, self).__init__()
        self.emb_size = emb_size
        self.drop_out = torch.nn.Dropout(drop_rate).to(device)
        self.device = device

        self.bn = torch.nn.BatchNorm1d(num_feature, momentum=bn_mom).to(device)
        self.if_separa = if_separa

        self.if_mlp_trans = if_mlp_trans
        # TRANS LINEAR
        if if_mlp_trans:
            self.trans_lin = nn.Linear(
                (un_layer + 1) * num_feature, (un_layer + 1) * emb_size, bias=if_trans_bias).to(device)
            if trans_init != 'none':
                if trans_init == 'xavier':
                    nn.init.xavier_uniform_(self.trans_lin.weight)
                elif trans_init == 'kaiming':
                    nn.init.kaiming_normal_(self.trans_lin.weight)
            self.trans_bn = torch.nn.BatchNorm1d((un_layer + 1) * emb_size, momentum=bn_mom).to(device)
        else:
            self.if_trans_share = if_trans_share
            if if_trans_share:
                self.trans_lin = nn.Linear(num_feature, emb_size, bias=if_trans_bias).to(device)
                if trans_init != 'none':
                    if trans_init == 'xavier':
                        nn.init.xavier_uniform_(self.trans_lin.weight)
                    elif trans_init == 'kaiming':
                        nn.init.kaiming_normal_(self.trans_lin.weight)
            else:
                self.trans_lin = nn.ModuleList()
                for i in range(un_layer+1):
                    linear = nn.Linear(num_feature, emb_size, bias=if_trans_bias).to(device)
                    if trans_init != 'none':
                        if trans_init == 'xavier':
                            nn.init.xavier_uniform_(linear.weight)
                        elif trans_init == 'kaiming':
                            nn.init.kaiming_normal_(linear.weight)
                    self.trans_lin.append(linear)
            self.if_trans_bn = if_trans_bn
            self.if_bn_share = if_bn_share
            if if_trans_bn:
                if if_bn_share:
                    self.trans_bn = torch.nn.BatchNorm1d(emb_size, momentum=bn_mom).to(device)
                else:
                    self.trans_bn = torch.nn.BatchNorm1d((un_layer+1) * emb_size, momentum=bn_mom).to(device)

        # Trans Activation
        if trans_act == 'sigmoid':
            self.trans_act = nn.Sigmoid()
        elif trans_act == 'relu':
            self.trans_act = nn.ReLU()
        elif trans_act == 'tanh':
            self.trans_act = nn.Tanh()
        elif trans_act == 'leaky':
            self.trans_act = nn.LeakyReLU()
        elif trans_act == 'none':
            self.trans_act = lambda x: x
        else:
            raise ValueError('Trans Act %s not defined' % trans_act)
        # MLP Activation
        if mlp_act == 'sigmoid':
            self.mlp_act = nn.Sigmoid()
        elif mlp_act == 'relu':
            self.mlp_act = nn.ReLU()
        elif mlp_act == 'tanh':
            self.mlp_act = nn.Tanh()
        elif mlp_act == 'leaky':
            self.mlp_act = nn.LeakyReLU()
        elif mlp_act == 'none':
            self.mlp_act = lambda x: x
        else:
            raise ValueError('Trans Act %s not defined' % trans_act)
        # MLP
        mlp_list = []
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = (un_layer+1)*emb_size
            else:
                pre_size = mlp_size

            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if mlp_init != 'none':
                if mlp_init == 'xavier':
                    nn.init.xavier_uniform_(linear.weight)
                elif mlp_init == 'kaiming':
                    nn.init.kaiming_normal_(linear.weight)
            mlp_list.append(linear)
            if if_mlp_bn:
                mlp_list.append(nn.BatchNorm1d(mlp_size, momentum=bn_mom).to(device))
            mlp_list.extend([
                self.mlp_act,
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = (un_layer+1)*emb_size
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, num_classes, bias=True).to(device)
        if mlp_init != 'none':
            if mlp_init == 'xavier':
                nn.init.xavier_uniform_(linear.weight)
            elif mlp_init == 'kaiming':
                nn.init.kaiming_normal_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)


    def forward(self, X):
        batch = X.shape[0]
        input_dim = X.shape[1]
        X = self.bn(X.reshape([batch * input_dim, -1])).contiguous().view([batch, input_dim, -1])
        X = self.drop_out(X)
        if self.if_mlp_trans:
            trans_x = self.trans_bn(self.trans_act(self.trans_lin(X.reshape([batch, -1]))))
        else:
            if self.if_trans_share:
                if not self.if_separa:
                    trans_x = self.trans_act(self.trans_lin(X))
                else:
                    trans_list=[]
                    for i in range(input_dim):

                        trans_list.append(self.trans_act(self.trans_lin(X[:, i, :])))
                    trans_x = torch.cat(trans_list, dim=-1)
            else:
                trans_list = []
                for i in range(input_dim):
                    trans_list.append(
                        self.trans_act(self.trans_lin[i](X[:, i, :])))
                trans_x = torch.cat(trans_list, dim=-1)
            if self.if_trans_bn:
                if self.if_bn_share:
                    trans_x = self.trans_bn(trans_x.reshape([batch * input_dim, -1])).reshape([batch, -1])
                else:
                    trans_x = self.trans_bn(trans_x.reshape([batch, -1]))

        trans_x = self.drop_out(trans_x)
        pred_x = self.mlp(trans_x)
        return pred_x
