from transformers import BertModel, BertTokenizer
import re, torch
import torch.nn as nn
import torch.nn.functional as F
from model.SGNN import sgnn


class PepGPL(nn.Module):
    def __init__(self, config):
        super(PepGPL, self).__init__()

        # max_len = config.max_len
        d_model = 1024
        self.pad_pep_len = config.pad_pep_len
        self.pad_pro_len = config.pad_pro_len
        self.device = config.device
        self.bert = BertModel.from_pretrained("../prot_bert_bfd")
        # self.pep_seq_fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())
        self.pep_seq_fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU())
        self.pep_seq_pooler_fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.pep_2_emb = nn.Embedding(7 + 1, 256, padding_idx=0)
        self.pep_ss3_emb = nn.Embedding(63 + 1, 256, padding_idx=0)
        self.pep_numerical_fc = nn.Sequential(nn.Linear(in_features=3, out_features=256), nn.ReLU())

        # only  use bridge model
        # self.pep_pool = torch.nn.MaxPool1d(kernel_size=config.pad_pep_len)
        # self.pro_pool = torch.nn.MaxPool1d(kernel_size=config.pad_pro_len)

        self.pep_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=512,
                      kernel_size=7,
                      stride=1,
                      padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=7,
                      stride=1,
                      padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.pad_pep_len)
        )

        self.pro_seq_fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU())
        # self.pro_seq_fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())
        self.pro_2_emb = nn.Embedding(7 + 1, 256, padding_idx=0)
        self.pro_ss3_emb = nn.Embedding(63 + 1, 256, padding_idx=0)
        self.pro_numerical_fc = nn.Linear(in_features=23, out_features=256)
        # use pro-cnn
        self.pro_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=512,
                      kernel_size=9,
                      stride=1,
                      padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=9,
                      stride=1,
                      padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.pad_pro_len)
        )

        self.block1 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Linear(1024, 1)
        )

        self.block2 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Linear(1024, 1)
        )

        self.sgnn = sgnn(config.device, outSize=512,
                         cHiddenSizeList=[1024],
                         fHiddenSizeList=[1024, 512], cSize=9723,
                         gcnHiddenSizeList=[512], fcHiddenSizeList=[512, 512], nodeNum=64,
                         hdnDropout=0.2, fcDropout=0.2)

    def forward(self, pep_encoded_input, pro_encoded_input, pep_dense, pro_dense, pep_ss3, pro_ss3, pep_2, pro_2):
        for key in pep_encoded_input:
            pep_encoded_input[key] = pep_encoded_input[key].to(self.device)
        for key in pro_encoded_input:
            pro_encoded_input[key] = pro_encoded_input[key].to(self.device)

        with torch.no_grad():
            pep_bert_output = self.bert(**pep_encoded_input)
            pro_bert_output = self.bert(**pro_encoded_input)

        pep_bert_last_hidden_output = pep_bert_output[0]
        pep_seq_output = self.pep_seq_fc(pep_bert_last_hidden_output[:, 1:-1, :])
        pep_2 = self.pep_2_emb(pep_2)
        pep_ss3 = self.pep_ss3_emb(pep_ss3)
        pep_dense = self.pep_numerical_fc(pep_dense)
        pep_feature = torch.cat([pep_seq_output, pep_ss3, pep_2, pep_dense], -1)

        pro_bert_last_hidden_output = pro_bert_output[0]
        pro_seq_output = self.pro_seq_fc(pro_bert_last_hidden_output[:, 1:-1, :])
        pro_2 = self.pro_2_emb(pro_2)
        pro_ss3 = self.pro_ss3_emb(pro_ss3)
        pro_dense = self.pro_numerical_fc(pro_dense)
        # pro_seq_emb = self.pro_cnn(pep_bert_output.last_hidden_state.permute(0, 2, 1))
        pep_total_feature = self.pep_cnn(pep_feature.permute(0, 2, 1)).squeeze(2)
        pro_feature = torch.cat([pro_seq_output, pro_ss3, pro_2, pro_dense], -1)
        pro_total_feature = self.pro_cnn(pro_feature.permute(0, 2, 1)).squeeze(2)

        pep_residue_feature = self.block1(pep_feature)
        pro_residue_feature = self.block2(pro_feature)
        pred_PepPI, pred_pep_bs, pred_prot_bs = self.sgnn(pep_total_feature, pro_total_feature, pep_residue_feature, pro_residue_feature)
        return pred_PepPI, pred_pep_bs, pred_prot_bs

        # pred_pep_bs = self.sgnn(pep_total_feature, pro_total_feature, pep_residue_feature,
        #                                                   pro_residue_feature)
        # return pred_pep_bs

