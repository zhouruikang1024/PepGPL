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
from torch.nn import functional as F
import pynvml

pynvml.nvmlInit()

# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def watch_nvidia(nvidia_ids, min_memory):
    flag = [1 for i in nvidia_ids]
    for i in nvidia_ids:
        # if i == 0 or i == 1 or i == 2:
        #     continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("card {} free memory is {}GB".format(i, meminfo.free * 1.0 / (1024 ** 3)))
        if meminfo.free * 1.0 / (1024 ** 3) > min_memory:
            flag[i] = 0
        else:
            flag[i] = 1
    if 0 in flag:
        free_card = []
        num = 0
        for i in flag:
            if i == 0:
                free_card.append(num)
            num = num + 1
        return free_card
    else:
        # print("no free card!")
        return []


def cls_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # r2_score, mean_squred_error are ignored
    return roc_auc_score(label, pred), average_precision_score(label, pred)


def random_split(self, fold=5):
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    for train_index, test_index in skf.split(self[0], self[11]):
        train_idx_list.append(train_index)
        val, test = [], []
        for val_index_inner, test_index_inner in ss.split(self[0][test_index], self[11][test_index]):
            for num, idx in enumerate(test_index):
                if num in val_index_inner:
                    val.append(idx)
                else:
                    test.append(idx)
        val_idx_list.append(val)
        test_idx_list.append(test)

    return train_idx_list, val_idx_list, test_idx_list


def boost_mask_BCE_loss(y_pred, y_true, pep_mask, bs_flag):
    loss = bs_flag * nn.BCELoss(reduction='none')(y_pred, y_true) * pep_mask
    return torch.sum(loss) / torch.sum(pep_mask)


def model_eval(dataloader, model, PepPI_criterion, config, val_flag):
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
    if val_flag:
        print(
            'validation - loss: {:.4f} | PepPI_auc: {:.3f} | PepPI_aupr: {:.3f} | PepPI_acc: {:.3f} | PepPI_prec: {:.3f} | PepPI_sen: {:.3f} | PepPI_spec: {:.3f} | PepPI_F1: {:.3f} | PepPI_MCC: {:.3f}'.format(
                avg_loss, PepPI_auc, PepPI_auprc, PepPI_metric[0], PepPI_metric[1],
                PepPI_metric[2], PepPI_metric[3], PepPI_metric[4], PepPI_metric[6]))

        print(
            'validation - loss: {:.4f} | pep_bs_auc: {:.3f} | pep_bs_auprc: {:.3f} | pep_bs_mcc: {:.3f} | pep_bs_acc: {:.3f} | pep_bs_prec: {:.3f} | pep_bs_sen: {:.3f} | pep_bs_spec: {:.3f} | pep_bs_F1: {:.3f}'.format(
                avg_loss, pep_bs_auc, pep_bs_auprc, pep_bs_mcc, pep_bs_metric[0], pep_bs_metric[1], pep_bs_metric[2],
                pep_bs_metric[3], pep_bs_metric[4]))
        print(
            'validation - loss: {:.4f} | pro_bs_auc: {:.3f} | pro_bs_auprc: {:.3f} | pro_bs_mcc: {:.3f} | pro_bs_acc: {:.3f} | pro_bs_prec: {:.3f} | pro_bs_sen: {:.3f} | pro_bs_spec: {:.3f} | pro_bs_F1: {:.3f}'.format(
                avg_loss, pro_bs_auc, pro_bs_auprc, pro_bs_mcc, pro_bs_metric[0], pro_bs_metric[1], pro_bs_metric[2],
                pro_bs_metric[3], pro_bs_metric[4]))
    else:
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
    return avg_loss, PepPI_auc, pep_bs_auc, pro_bs_auc


def periodic_test(test_iter, model, criterion, config, val_flag):
    test_loss, PepPI_auc, pep_bs_auc, pro_bs_auc = model_eval(test_iter, model, criterion, config, val_flag)
    performance = [PepPI_auc, pep_bs_auc, pro_bs_auc]

    return performance


def train(train_iter, valid_iter, test_iter, model, optimizer, PepPI_criterion, config, iter_k):
    best_PepPI_auc = 0
    best_pep_bs_acc = 0
    best_pep_bs_auc = 0
    best_pro_bs_auc = 0
    best_performance = []
    for epoch in range(1, config.epoch + 1):
        steps = 0
        train_epoch_loss = 0
        PepPI_correct_num = 0
        pep_bs_correct_num = 0
        pro_bs_correct_num = 0
        train_PepPI_total_num = 0
        train_pep_bs_total_num = 0
        train_pro_bs_total_num = 0

        label_PepPI_pred = torch.empty([0]).to(config.device)
        label_PepPI_real = torch.empty([0]).to(config.device)
        pred_PepPI_prob = torch.empty([0]).to(config.device)

        label_pep_bs_pred = torch.empty([0]).to(config.device)
        label_pep_bs_real = torch.empty([0]).to(config.device)
        pred_pep_bs_prob = torch.empty([0]).to(config.device)

        label_pro_bs_pred = torch.empty([0]).to(config.device)
        label_pro_bs_real = torch.empty([0]).to(config.device)
        pred_pro_bs_prob = torch.empty([0]).to(config.device)

        # epoch_last_hidden_state = torch.empty([0]).cuda()
        # epoch_pred_bs = torch.empty([0]).cuda()
        model.train()
        for batch in train_iter:
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

            pep_ss3 = pep_ss3.long().to(config.device)
            pro_ss3 = pro_ss3.long().to(config.device)
            pep_2 = pep_2[:, :].long().to(config.device)
            pro_2 = pro_2[:, :].long().to(config.device)
            pep_dense = pep_dense.float().to(config.device)
            pro_dense = pro_dense.float().to(config.device)
            bs_flag = bs_flag.to(config.device)
            label = label.reshape(-1).float().to(config.device)
            pep_bs = pep_bs.float().to(config.device)
            prot_bs_flag = prot_bs_flag.to(config.device)
            prot_bs = prot_bs.float().to(config.device)

            pred_PepPI, pred_pep_bs, pred_prot_bs = model(pep_encoded_input, pro_encoded_input, pep_dense, pro_dense,
                                                          pep_ss3,
                                                          pro_ss3, pep_2, pro_2)

            pred_pep_bs = pred_pep_bs[:, :]
            pep_mask = pep_mask[:, :].to(config.device)
            pep_mask_no_pad = torch.zeros(config.batch_size, config.pad_pep_len).long().to(config.device)
            for row_num, row in enumerate(pep_mask):
                pep_mask_no_pad[row_num, 0:torch.sum(row) - 2] = row[1:torch.sum(row) - 1]

            pred_prot_bs = pred_prot_bs[:, :]
            pro_mask = pro_mask[:, :].to(config.device)
            pro_mask_no_pad = torch.zeros(config.batch_size, config.pad_pro_len).long().cuda()
            for row_num, row in enumerate(pro_mask):
                pro_mask_no_pad[row_num, 0:torch.sum(row) - 2] = row[1:torch.sum(row) - 1]

            PepPI_loss = PepPI_criterion(pred_PepPI, label)
            pep_residue_loss = boost_mask_BCE_loss(pred_pep_bs, pep_bs, pep_mask_no_pad, bs_flag)
            prot_residue_loss = boost_mask_BCE_loss(pred_prot_bs, prot_bs, pro_mask_no_pad, prot_bs_flag)
            loss = PepPI_loss + pep_residue_loss + 10 * prot_residue_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps = steps + 1
            train_batch_loss = loss.item()
            train_epoch_loss = train_epoch_loss + train_batch_loss

            pred_PepPI_class = (pred_PepPI > 0.5).int()
            label_PepPI_pred = torch.cat([label_PepPI_pred, pred_PepPI_class.float()])
            label_PepPI_real = torch.cat([label_PepPI_real, label.float()])
            pred_PepPI_prob = torch.cat([pred_PepPI_prob, pred_PepPI])

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
            train_pep_bs_total_num = train_pep_bs_total_num + batch_pep_bs_num

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
            train_pro_bs_total_num = train_pro_bs_total_num + batch_pro_bs_num

            pred_pro_bs_class = (pred_pro_bs_real_len > 0.5).int()
            label_pro_bs_pred = torch.cat([label_pro_bs_pred, pred_pro_bs_class.float().reshape(-1)])
            label_pro_bs_real = torch.cat([label_pro_bs_real, real_pro_bs.reshape(-1)])
            pred_pro_bs_prob = torch.cat([pred_pro_bs_prob, pred_pro_bs_real_len.reshape(-1)])

        sum_epoch = epoch
        PepPI_metric, PepPI_roc_data, PepPI_prc_data = util_metric.caculate_metric(
            label_PepPI_pred.cpu().detach().numpy(), label_PepPI_real.cpu().detach().numpy(),
            pred_PepPI_prob.cpu().detach().numpy())
        PepPI_auc = PepPI_roc_data[2]
        PepPI_auprc = PepPI_prc_data[2]

        pep_bs_metric, pep_bs_roc_data, pep_bs_prc_data = util_metric.caculate_metric(
            label_pep_bs_pred.cpu().detach().numpy(), label_pep_bs_real.cpu().detach().numpy(),
            pred_pep_bs_prob.cpu().detach().numpy())
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
            'Train - Epoch{} - loss: {:.4f} | PepPI_auc: {:.3f} | PepPI_aupr: {:.3f} | PepPI_acc: {:.3f} | PepPI_prec: {:.3f} | PepPI_sen: {:.3f} | PepPI_spec: {:.3f} | PepPI_F1: {:.3f} | PepPI_MCC: {:.3f}'.format(
                sum_epoch, train_epoch_loss / (len(train_iter)), PepPI_auc, PepPI_auprc, PepPI_metric[0],
                PepPI_metric[1], PepPI_metric[2], PepPI_metric[3], PepPI_metric[4], PepPI_metric[6]))

        print(
            'Train - Epoch{} - loss: {:.4f} | pep_bs_auc: {:.3f} | pep_bs_aupr: {:.3f} | pep_bs_mcc: {:.3f} | pep_bs_acc: {:.3f} | pep_bs_prec: {:.3f} | pep_bs_sen: {:.3f} | pep_bs_spec: {:.3f} | pep_bs_F1: {:.3f}'.format(
                sum_epoch, train_epoch_loss / (len(train_iter)), pep_bs_auc, pep_bs_auprc,
                pep_bs_mcc, pep_bs_metric[0], pep_bs_metric[1], pep_bs_metric[2], pep_bs_metric[3], pep_bs_metric[4]))
        print(
            'Train - Epoch{} - loss: {:.4f} | pro_bs_auc: {:.3f} | pro_bs_aupr: {:.3f} | pro_bs_mcc: {:.3f} | pro_bs_acc: {:.3f} | pro_bs_prec: {:.3f} | pro_bs_sen: {:.3f} | pro_bs_spec: {:.3f} | pro_bs_F1: {:.3f}'.format(
                sum_epoch, train_epoch_loss / (len(train_iter)), pro_bs_auc, pro_bs_auprc,
                pro_bs_mcc, pro_bs_metric[0], pro_bs_metric[1], pro_bs_metric[2], pro_bs_metric[3], pro_bs_metric[4]))

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:

            periodic_test(valid_iter, model, PepPI_criterion, config, True)
            periodic_test(test_iter, model, PepPI_criterion, config, False)


def train_test(train_iter, val_iter, test_iter, config, iter_k):
    # 加载
    model = Network.PepGPL(config)
    if config.cuda:
        model.to(config.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)

    PepPI_criterion = nn.BCELoss()

    print('=' * 50 + 'Start Training' + '=' * 50)
    train(train_iter, val_iter, test_iter, model, optimizer, PepPI_criterion, config, iter_k)
    print('=' * 50 + 'Train Finished' + '=' * 50)


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)

    '''load configuration'''
    config = cf.get_train_config()

    '''set device'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    '''load data'''
    data_iter = load_data(config)
    print('=' * 20, 'load data over', '=' * 20)


    '''train procedure'''
    valid_performance = 0
    test_performance = np.array([])
    last_test_metric = 0

    train_idx_list, val_idx_list, test_idx_list = random_split(data_iter, config.k_fold)

    test_loss_list, test_PepPI_auc_list, test_PepPI_auprc_list = [], [], []
    test_residue_auc_list, test_residue_auprc_list = [], []
    for fold in range(config.k_fold):
        print('fold', fold + 1, 'begin training')
        train_ind, val_ind, test_ind = train_idx_list[fold], val_idx_list[fold], test_idx_list[fold]
        train_dataset = TensorDataset(torch.from_numpy(data_iter[0][train_ind]),
                                      torch.from_numpy(data_iter[1][train_ind]),
                                      torch.from_numpy(data_iter[2][train_ind]),
                                      torch.from_numpy(data_iter[3][train_ind]),
                                      torch.from_numpy(data_iter[4][train_ind]),
                                      torch.from_numpy(data_iter[5][train_ind]),
                                      torch.from_numpy(data_iter[6][train_ind]),
                                      torch.from_numpy(data_iter[7][train_ind]),
                                      torch.from_numpy(data_iter[8][train_ind]),
                                      torch.from_numpy(data_iter[9][train_ind]),
                                      torch.from_numpy(data_iter[10][train_ind]),
                                      torch.from_numpy(data_iter[11][train_ind]),
                                      torch.from_numpy(data_iter[12][train_ind]),
                                      torch.from_numpy(data_iter[13][train_ind]),
                                      torch.from_numpy(data_iter[14][train_ind])
                                      )

        val_dataset = TensorDataset(torch.from_numpy(data_iter[0][val_ind]),
                                     torch.from_numpy(data_iter[1][val_ind]),
                                     torch.from_numpy(data_iter[2][val_ind]),
                                     torch.from_numpy(data_iter[3][val_ind]),
                                     torch.from_numpy(data_iter[4][val_ind]),
                                     torch.from_numpy(data_iter[5][val_ind]),
                                     torch.from_numpy(data_iter[6][val_ind]),
                                     torch.from_numpy(data_iter[7][val_ind]),
                                     torch.from_numpy(data_iter[8][val_ind]),
                                     torch.from_numpy(data_iter[9][val_ind]),
                                     torch.from_numpy(data_iter[10][val_ind]),
                                     torch.from_numpy(data_iter[11][val_ind]),
                                     torch.from_numpy(data_iter[12][val_ind]),
                                     torch.from_numpy(data_iter[13][val_ind]),
                                     torch.from_numpy(data_iter[14][val_ind])
                                     )

        test_dataset = TensorDataset(torch.from_numpy(data_iter[0][test_ind]),
                                     torch.from_numpy(data_iter[1][test_ind]),
                                     torch.from_numpy(data_iter[2][test_ind]),
                                     torch.from_numpy(data_iter[3][test_ind]),
                                     torch.from_numpy(data_iter[4][test_ind]),
                                     torch.from_numpy(data_iter[5][test_ind]),
                                     torch.from_numpy(data_iter[6][test_ind]),
                                     torch.from_numpy(data_iter[7][test_ind]),
                                     torch.from_numpy(data_iter[8][test_ind]),
                                     torch.from_numpy(data_iter[9][test_ind]),
                                     torch.from_numpy(data_iter[10][test_ind]),
                                     torch.from_numpy(data_iter[11][test_ind]),
                                     torch.from_numpy(data_iter[12][test_ind]),
                                     torch.from_numpy(data_iter[13][test_ind]),
                                     torch.from_numpy(data_iter[14][test_ind])
                                     )

        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

        train_test(train_loader, val_loader, test_loader, config, fold)

