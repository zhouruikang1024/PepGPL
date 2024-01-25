from torch import nn as nn
from torch.nn import functional as F
import torch, time, os, random
import numpy as np
from collections import OrderedDict


class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(embedding, dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout

    def forward(self, x):
        # x: batchSize × seqLen
        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x



class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False,
                 outAct=False, outDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens, bns = [], []
        for i, os in enumerate(hiddenList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ))
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.last_actFunc = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        for h, bn in zip(self.hiddens, self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape) == 2 else bn(x.transpose(-1, -2)).transpose(-1, -2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn:
            x = self.bns[-1](x) if len(x.shape) == 2 else self.bns[-1](x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct:
            x = self.actFunc(x)
        else:
            x = self.last_actFunc(x)
        if self.outDp:
            x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False,
                 outBn=False, outAct=False, outDp=False, resnet=False, name='GCN', actFunc=nn.ReLU):
        super(GCN, self).__init__()
        self.name = name
        hiddens, bns = [], []
        for i, os in enumerate(hiddenSizeList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ))
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet

    def forward(self, x, L):
        # x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for h, bn in zip(self.hiddens, self.bns):
            a = h(torch.matmul(L, x))  # => batchSize × nodeNum × os
            if self.bnEveryLayer:
                if len(L.shape) == 3:
                    a = bn(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = bn(a)
            a = self.actFunc(a)
            if self.dpEveryLayer:
                a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a
        a = self.out(torch.matmul(L, x))  # => batchSize × nodeNum × outSize
        if self.outBn:
            if len(L.shape) == 3:
                a = self.bns[-1](a.transpose(1, 2)).transpose(1, 2)
            else:
                a = self.bns[-1](a)
        if self.outAct:
            a = self.actFunc(a)
        if self.outDp:
            a = self.dropout(a)
        if self.resnet and a.shape == x.shape:
            a += x
        x = a
        return x
