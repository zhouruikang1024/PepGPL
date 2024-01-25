import torch

from model.nnLayer import *


class sgnn(nn.Module):
    def __init__(self, device, outSize,
                 cHiddenSizeList,
                 fHiddenSizeList, cSize=9723,
                 gcnHiddenSizeList=[], fcHiddenSizeList=[], nodeNum=32, resnet=True,
                 hdnDropout=0.1, fcDropout=0.2, ):
        super(sgnn, self).__init__()
        self.device = device
        self.nodeEmbedding = TextEmbedding(
            torch.tensor(np.random.normal(size=(max(nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout,
            name='nodeEmbedding')

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout, dpEveryLayer=True,
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet)
        #
        # self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True, dpEveryLayer=True).to(
        #     device)
        self.pep_bs_fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True,
                                   dpEveryLayer=True, outDp=False)

        self.prot_bs_fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True,
                                    dpEveryLayer=True, outDp=False)

        self.pep_all_feature_Linear = MLP(512, outSize, fcHiddenSizeList, dropout=fcDropout, outAct=True, outDp=True,
                                          bnEveryLayer=True, dpEveryLayer=True, outBn=True)
        self.pro_all_feature_Linear = MLP(512, outSize, fcHiddenSizeList, dropout=fcDropout, outAct=True,
                                          bnEveryLayer=True, dpEveryLayer=True)
        # self.criterion = nn.BCEWithLogitsLoss()
        #
        # self.embModuleList = nn.ModuleList([])
        # self.finetunedEmbList = nn.ModuleList([])
        # self.moduleList = nn.ModuleList(
        #     [self.nodeEmbedding, self.cFcLinear, self.fFcLinear, self.nodeGCN, self.fcLinear, self.pepFcLinear,
        #      self.proFcLinear])
        # self.sampleType = sampleType

        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout

        self.last_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # self.block1 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )

    def forward(self, pep_feature, pro_feature, pep_residue_feature, pro_residue_feature):
        Xpep = self.pep_all_feature_Linear(pep_feature).unsqueeze(1)
        Xpro = self.pro_all_feature_Linear(pro_feature).unsqueeze(1)
        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(Xpro), 1, 1)
            node = torch.cat([Xpep, Xpro, pep_residue_feature, pro_residue_feature, node], dim=1)
            # node = torch.cat([Xpep, Xpro, pep_residue_feature, node], dim=1)
            # node = torch.cat([Xpep, Xpro, node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)

            cosNode = torch.matmul(node, node.transpose(1, 2)) / (
                    nodeDist * nodeDist.transpose(1, 2) + 1e-8)

            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            cosNode = F.relu(cosNode)  # => batchSize × nodeNum × nodeNum
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(Xpep), 1,
                                                                                         1)
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)
            node_gcned = self.nodeGCN(node, pL)

            node_embed = node_gcned[:, 0, :] * node_gcned[:, 1, :]
            prot_residue_embed = node_gcned[:, 52:728, :] * node_gcned[:, 0, :].unsqueeze(1).repeat(1, 676, 1)
            pep_residue_embed = node_gcned[:, 2:52, :] * node_gcned[:, 1, :].unsqueeze(1).repeat(1, 50, 1)
            return torch.sigmoid(self.last_fc(node_embed).squeeze(dim=1)), self.pep_bs_fcLinear(pep_residue_embed).squeeze(2), self.prot_bs_fcLinear(prot_residue_embed).squeeze(2)

            # only pep_bs prediction
            # pep_residue_embed = node_gcned[:, 2:52, :] * node_gcned[:, 1, :].unsqueeze(1).repeat(1, 50, 1)
            # return self.pep_bs_fcLinear(pep_residue_embed).squeeze(2)
