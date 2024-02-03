import os
import sys

from sklearn.metrics import roc_auc_score, average_precision_score

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config as cf
from utils.data_loader_protBert import *
from utils import util_metric
from model import Network
import numpy as np

import time
import torch.backends.cudnn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import random
import dill
import pandas as pd


amino_acid_set = {5: "L", 6: "A", 7: "G", 8: "V", 9: "E", 10: "S", 11: "I", 12: "K", 13: "R", 14: "D", 15: "T",
                  16: "P", 17: "N", 18: "Q", 19: "F", 20: "Y", 21: "M", 22: "H", 23: "C", 24: "W", 25: "X"}


def cls_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # r2_score, mean_squred_error are ignored
    return roc_auc_score(label, pred), average_precision_score(label, pred)


def boost_mask_BCE_loss(y_pred, y_true, pep_mask, bs_flag):
    loss = bs_flag * nn.BCELoss(reduction='none')(y_pred, y_true) * pep_mask
    return torch.sum(loss) / torch.sum(pep_mask)


def model_eval(dataloader, model, PepPI_criterion, config):
    label_PepPI_pred = torch.empty([0]).to(config.device)
    label_PepPI_real = torch.empty([0]).to(config.device)
    pred_PepPI_prob = torch.empty([0]).to(config.device)

    label_pep_bs_pred = torch.empty([0]).to(config.device)
    label_pep_bs_real = torch.empty([0]).to(config.device)
    pred_pep_bs_prob = torch.empty([0]).to(config.device)

    label_pro_bs_pred = torch.empty([0]).to(config.device)
    label_pro_bs_real = torch.empty([0]).to(config.device)
    pred_pro_bs_prob = torch.empty([0]).to(config.device)

    # print('model_eval dataloader', len(dataloader))

    iter_size, corrects, pep_bs_iter_size, pep_bs_correct_num, avg_loss = 0, 0, 0, 0, 0
    pro_bs_correct_num = 0
    test_pro_bs_total_num = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pep_encoded_input = {}
            pro_encoded_input = {}

            pep_seq, pro_seq, pep_mask, pro_mask, pep_ss3, pro_ss3, pep_dense, pro_dense, pep_2, pro_2, bs_flag, label, pep_bs, prot_bs_flag, prot_bs = batch
            pep_encoded_input['input_ids'] = pep_seq
            pep_encoded_input['token_type_ids'] = torch.zeros([config.batch_size, config.pad_pep_len + 2],
                                                              dtype=torch.long)
            pep_encoded_input['attention_mask'] = pep_mask
            pro_encoded_input['input_ids'] = pro_seq
            pro_encoded_input['token_type_ids'] = torch.zeros([config.batch_size, config.pad_pro_len + 2],
                                                              dtype=torch.long)
            pro_encoded_input['attention_mask'] = pro_mask
            pep_2 = pep_2[:, :].long().to(config.device)
            pro_2 = pro_2[:, :].long().to(config.device)
            pep_ss3 = pep_ss3.long().to(config.device)
            pro_ss3 = pro_ss3.long().to(config.device)
            pep_dense = pep_dense.float().to(config.device)
            pro_dense = pro_dense.float().to(config.device)
            bs_flag = bs_flag.to(config.device)
            label = label.reshape(-1).float().to(config.device)
            pep_bs = pep_bs.float().to(config.device)
            prot_bs_flag = prot_bs_flag.to(config.device)
            prot_bs = prot_bs.float().to(config.device)

            pred_PepPI, pred_pep_bs, pred_prot_bs = model(pep_encoded_input, pro_encoded_input, pep_dense, pro_dense,
                                                          pep_ss3, pro_ss3, pep_2,
                                                          pro_2)

            pred_pep_bs = pred_pep_bs[:, :]
            pep_mask = pep_mask[:, :].to(config.device)
            pep_mask_no_pad = torch.zeros(config.batch_size, config.pad_pep_len).long().to(config.device)
            for row_num, row in enumerate(pep_mask):
                pep_mask_no_pad[row_num, 0:torch.sum(row) - 2] = row[1:torch.sum(row) - 1]

            pred_prot_bs = pred_prot_bs[:, :]
            pro_mask = pro_mask[:, :].to(config.device)
            pro_mask_no_pad = torch.zeros(config.batch_size, config.pad_pro_len).long().to(config.device)
            for row_num, row in enumerate(pro_mask):
                pro_mask_no_pad[row_num, 0:torch.sum(row) - 2] = row[1:torch.sum(row) - 1]

            PepPI_loss = PepPI_criterion(pred_PepPI, label)
            pep_residue_loss = boost_mask_BCE_loss(pred_pep_bs, pep_bs, pep_mask_no_pad, bs_flag)
            prot_residue_loss = boost_mask_BCE_loss(pred_prot_bs, prot_bs, pro_mask_no_pad, prot_bs_flag)
            loss = PepPI_loss + pep_residue_loss + 10 * prot_residue_loss
            # loss = PepPI_loss

            avg_loss = avg_loss + loss.item()

            pred_PepPI_class = (pred_PepPI > 0.5).int()
            label_PepPI_pred = torch.cat([label_PepPI_pred, pred_PepPI_class.float()])
            label_PepPI_real = torch.cat([label_PepPI_real, label.float()])
            pred_PepPI_prob = torch.cat([pred_PepPI_prob, pred_PepPI])

            PepPI_corre = ((pred_PepPI > 0.5).int() == label).int()
            PepPI_corrects = PepPI_corre.sum()
            corrects = corrects + PepPI_corrects
            the_batch_size = label.size(0)
            iter_size = iter_size + the_batch_size

            pred_pep_bs_real_len = torch.empty([0]).to(config.device)
            real_pep_bs = torch.empty([0]).to(config.device)
            for i, pep_len in enumerate(pep_mask_no_pad):
                if bs_flag[i] != 0:
                    pred_pep_bs_real_len = torch.cat(
                        (pred_pep_bs_real_len, (pred_pep_bs[i, :torch.sum(pep_len)]).reshape(-1)))
                    real_pep_bs = torch.cat((real_pep_bs, (pep_bs[i, :torch.sum(pep_len)]).reshape(-1)))
            pep_bs_corre = ((pred_pep_bs_real_len > 0.5).int() == real_pep_bs).int()
            pep_bs_corrects = pep_bs_corre.sum()
            pep_bs_correct_num = pep_bs_correct_num + pep_bs_corrects
            batch_pep_bs_num = real_pep_bs.size(0)
            pep_bs_iter_size = pep_bs_iter_size + batch_pep_bs_num

            pred_pep_bs_class = (pred_pep_bs_real_len > 0.5).int()
            label_pep_bs_pred = torch.cat([label_pep_bs_pred, pred_pep_bs_class.float().reshape(-1)])
            label_pep_bs_real = torch.cat([label_pep_bs_real, real_pep_bs.reshape(-1)])
            pred_pep_bs_prob = torch.cat([pred_pep_bs_prob, pred_pep_bs_real_len.reshape(-1)])

            pred_pro_bs_real_len = torch.empty([0]).to(config.device)
            real_pro_bs = torch.empty([0]).to(config.device)
            for i, pro_len in enumerate(pro_mask_no_pad):
                if prot_bs_flag[i] != 0:
                    pred_pro_bs_real_len = torch.cat(
                        (pred_pro_bs_real_len, (pred_prot_bs[i, :torch.sum(pro_len)]).reshape(-1)))
                    real_pro_bs = torch.cat((real_pro_bs, (prot_bs[i, :torch.sum(pro_len)]).reshape(-1)))

            pro_bs_corre = ((pred_pro_bs_real_len > 0.5).int() == real_pro_bs).int()
            pro_bs_corrects = pro_bs_corre.sum()
            pro_bs_correct_num = pro_bs_correct_num + pro_bs_corrects
            batch_pro_bs_num = real_pro_bs.size(0)
            test_pro_bs_total_num = test_pro_bs_total_num + batch_pro_bs_num

            pred_pro_bs_class = (pred_pro_bs_real_len > 0.5).int()
            label_pro_bs_pred = torch.cat([label_pro_bs_pred, pred_pro_bs_class.float().reshape(-1)])
            label_pro_bs_real = torch.cat([label_pro_bs_real, real_pro_bs.reshape(-1)])
            pred_pro_bs_prob = torch.cat([pred_pro_bs_prob, pred_pro_bs_real_len.reshape(-1)])

    PepPI_metric, PepPI_roc_data, PepPI_prc_data = util_metric.caculate_metric(label_PepPI_pred.cpu().numpy(),
                                                                               label_PepPI_real.cpu().numpy(),
                                                                               pred_PepPI_prob.cpu().numpy())
    avg_loss = avg_loss / len(dataloader)
    # accuracy = 100.0 * corrects / iter_size
    PepPI_auc = PepPI_roc_data[2]
    PepPI_auprc = PepPI_prc_data[2]

    pep_bs_metric, pep_bs_roc_data, pep_bs_prc_data = util_metric.caculate_metric(label_pep_bs_pred.cpu().numpy(),
                                                                                  label_pep_bs_real.cpu().numpy(),
                                                                                  pred_pep_bs_prob.cpu().numpy())

    pep_bs_auc = pep_bs_roc_data[2]
    pep_bs_auprc = pep_bs_prc_data[2]
    pep_bs_mcc = pep_bs_metric[6]

    pro_bs_metric, pro_bs_roc_data, pro_bs_prc_data = util_metric.caculate_metric(
        label_pro_bs_pred.cpu().detach().numpy(), label_pro_bs_real.cpu().detach().numpy(),
        pred_pro_bs_prob.cpu().detach().numpy())
    pro_bs_auc = pro_bs_roc_data[2]
    pro_bs_auprc = pro_bs_prc_data[2]
    pro_bs_mcc = pro_bs_metric[6]

    print(
        'test - loss: {:.4f} | PepPI_auc: {:.3f} | PepPI_aupr: {:.3f} | PepPI_acc: {:.3f} | PepPI_prec: {:.3f} | PepPI_sen: {:.3f} | PepPI_spec: {:.3f} | PepPI_F1: {:.3f} | PepPI_MCC: {:.3f}'.format(
            avg_loss, PepPI_auc, PepPI_auprc, PepPI_metric[0], PepPI_metric[1],
            PepPI_metric[2], PepPI_metric[3], PepPI_metric[4], PepPI_metric[6]))
    print(
        'test - loss: {:.4f} | pep_bs_auc: {:.3f} | pep_bs_auprc: {:.3f} | pep_bs_mcc: {:.3f} | pep_bs_acc: {:.3f} | pep_bs_prec: {:.3f} | pep_bs_sen: {:.3f} | pep_bs_spec: {:.3f} | pep_bs_F1: {:.3f}'.format(
            avg_loss, pep_bs_auc, pep_bs_auprc, pep_bs_mcc, pep_bs_metric[0], pep_bs_metric[1], pep_bs_metric[2],
            pep_bs_metric[3], pep_bs_metric[4]))
    print(
        'test - loss: {:.4f} | pro_bs_auc: {:.3f} | pro_bs_auprc: {:.3f} | pro_bs_mcc: {:.3f} | pro_bs_acc: {:.3f} | pro_bs_prec: {:.3f} | pro_bs_sen: {:.3f} | pro_bs_spec: {:.3f} | pro_bs_F1: {:.3f}'.format(
            avg_loss, pro_bs_auc, pro_bs_auprc, pro_bs_mcc, pro_bs_metric[0], pro_bs_metric[1], pro_bs_metric[2],
            pro_bs_metric[3], pro_bs_metric[4]))


def train_test(config, test_iter):
    # 加载
    model = Network.PepGPL(config)
    model.load_state_dict(torch.load("../result/model_state_dict.pth"))
    if config.cuda:
        model.to(config.device)
    PepPI_criterion = nn.BCELoss()

    print('=' * 50 + 'Start Testing' + '=' * 50)
    model_eval(test_iter, model, PepPI_criterion, config)
    print('=' * 50 + 'Train Finished' + '=' * 50)


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)

    '''load configuration'''
    config = cf.get_train_config()

    '''train procedure'''
    valid_performance = 0
    test_performance = np.array([])
    last_test_metric = 0

    test_loss_list, test_PepPI_auc_list, test_PepPI_auprc_list = [], [], []
    test_residue_auc_list, test_residue_auprc_list = [], []

    with open('../result/test_dataset.pkl', 'rb') as f:
        test_dataset = dill.load(f)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
                                    drop_last=True)

    train_test(config, test_loader)



