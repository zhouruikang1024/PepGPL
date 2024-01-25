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


def model_eval(data_iter, model, PepPI_criterion, config):
    pep_list = []
    prot_list = []
    PepPI_label_list = []
    pred_PepPI_list = []
    pep_bs_label_list = []
    pred_pep_bs_list = []
    prot_bs_label_list = []
    pred_prot_bs_list = []
    label_PepPI_pred = torch.empty([0]).cuda()
    label_PepPI_real = torch.empty([0]).cuda()
    pred_PepPI_prob = torch.empty([0]).cuda()

    label_pep_bs_pred = torch.empty([0]).cuda()
    label_pep_bs_real = torch.empty([0]).cuda()
    pred_pep_bs_prob = torch.empty([0]).cuda()

    label_pro_bs_pred = torch.empty([0]).cuda()
    label_pro_bs_real = torch.empty([0]).cuda()
    pred_pro_bs_prob = torch.empty([0]).cuda()

    # print('model_eval dataloader', len(dataloader))
    df_predict = pd.DataFrame(columns=['pep_seq', 'prot_seq', 'PepPI_label', 'pred_PepPI', 'pep_bs_label', 'pred_pep_bs', 'prot_bs_label', 'pred_prot_bs'])
    iter_size, corrects, pep_bs_iter_size, pep_bs_correct_num, avg_loss = 0, 0, 0, 0, 0
    pro_bs_correct_num = 0
    test_pro_bs_total_num = 0
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
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
            pep_2 = pep_2[:, :].cuda().long()
            pro_2 = pro_2[:, :].cuda().long()
            pep_ss3 = pep_ss3.cuda().long()
            pro_ss3 = pro_ss3.cuda().long()
            pep_dense = pep_dense.cuda().float()
            pro_dense = pro_dense.cuda().float()
            bs_flag = bs_flag.cuda()
            label = label.cuda().reshape(-1).float()
            pep_bs = pep_bs.cuda().float()
            prot_bs_flag = prot_bs_flag.cuda()
            prot_bs = prot_bs.cuda().float()

            pred_PepPI, pred_pep_bs, pred_prot_bs = model(pep_encoded_input, pro_encoded_input, pep_dense, pro_dense,
                                                          pep_ss3, pro_ss3, pep_2, pro_2)

            pred_pep_bs = pred_pep_bs[:, :]
            pep_mask = pep_mask[:, :].cuda()
            pep_mask_no_pad = torch.zeros(config.batch_size, 50).long().cuda()
            for row_num, row in enumerate(pep_mask):
                pep_mask_no_pad[row_num, 0:torch.sum(row) - 2] = row[1:torch.sum(row) - 1]

            pred_prot_bs = pred_prot_bs[:, :]
            pro_mask = pro_mask[:, :].cuda()
            pro_mask_no_pad = torch.zeros(config.batch_size, 676).long().cuda()
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

            pred_pep_bs_real_len = torch.empty([0]).cuda()
            real_pep_bs = torch.empty([0]).cuda()
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

            pred_pro_bs_real_len = torch.empty([0]).cuda()
            real_pro_bs = torch.empty([0]).cuda()
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


            PepPI_label = label.cpu().numpy()
            pred_PepPI_class = pred_PepPI_class.cpu().numpy()
            for row, peptide in enumerate(pep_seq):
                peptide = peptide[1:]
                cur_pep_seq = ''
                for res in peptide:
                    res = res.item()
                    if res != 3:
                        residue = amino_acid_set[res]
                        cur_pep_seq = cur_pep_seq + residue
                    else:
                        break
                pep_list.append(cur_pep_seq)
                protein = pro_seq[row, 1:]
                cur_pro_seq = ''
                for res in protein:
                    res = res.item()
                    if res != 3:
                        residue = amino_acid_set[res]
                        cur_pro_seq = cur_pro_seq + residue
                    else:
                        break
                prot_list.append(cur_pro_seq)
                PepPI_label_list.append(str(int(PepPI_label[row])))
                pred_PepPI_list.append(str(pred_PepPI_class[row]))

                cur_pep_seq_label = pep_bs[row, :len(cur_pep_seq)]
                cur_pep_seq_label = cur_pep_seq_label.cpu().numpy().tolist()
                cur_pep_seq_label = ''.join([str(int(x)) for x in cur_pep_seq_label])
                pep_bs_label_list.append(cur_pep_seq_label)

                cur_pep_seq_pred = pred_pep_bs[row, :len(cur_pep_seq)]
                cur_pep_seq_pred = (cur_pep_seq_pred > 0.5).int()
                cur_pep_seq_pred = cur_pep_seq_pred.detach().cpu().numpy().tolist()
                cur_pep_seq_pred = ''.join([str(int(x)) for x in cur_pep_seq_pred])
                pred_pep_bs_list.append(cur_pep_seq_pred)

                cur_prot_bs_label = prot_bs[row, :len(cur_pro_seq)]
                cur_prot_bs_label = cur_prot_bs_label.cpu().numpy().tolist()
                cur_prot_bs_label = ''.join([str(int(x)) for x in cur_prot_bs_label])
                prot_bs_label_list.append(cur_prot_bs_label)

                cur_prot_bs_pred = pred_prot_bs[row, :len(cur_pro_seq)]
                cur_prot_bs_pred = (cur_prot_bs_pred > 0.5).int()
                cur_prot_bs_pred = cur_prot_bs_pred.detach().cpu().numpy().tolist()
                cur_prot_bs_pred = ''.join([str(int(x)) for x in cur_prot_bs_pred])
                pred_prot_bs_list.append(cur_prot_bs_pred)

    for row, _ in enumerate(pep_list):
        df_predict = df_predict.append(
            pd.DataFrame([[pep_list[row], prot_list[row], PepPI_label_list[row], pred_PepPI_list[row],
                           pep_bs_label_list[row], pred_pep_bs_list[row], prot_bs_label_list[row],
                           pred_prot_bs_list[row]]],
                         columns=['pep_seq', 'prot_seq', 'PepPI_label', 'pred_PepPI', 'pep_bs_label', 'pred_pep_bs',
                                  'prot_bs_label',
                                  'pred_prot_bs']), ignore_index=True)
    df_predict.to_csv("../result/train_validation_test_result_1_1_10_shuffle_new/cv5_fold2_result.csv", encoding='utf-8', index=False, sep='\t')
    PepPI_metric, PepPI_roc_data, PepPI_prc_data = util_metric.caculate_metric(label_PepPI_pred.cpu().numpy(),
                                                                               label_PepPI_real.cpu().numpy(),
                                                                               pred_PepPI_prob.cpu().numpy())
    avg_loss = avg_loss / len(data_iter)
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
        'Evaluation - loss: {:.4f} | PepPI_auc: {:.3f} | PepPI_aupr: {:.3f} | PepPI_acc: {:.3f} | PepPI_prec: {:.3f} | PepPI_sen: {:.3f} | PepPI_spec: {:.3f} | PepPI_F1: {:.3f} | PepPI_MCC: {:.3f}'.format(
            avg_loss, PepPI_auc, PepPI_auprc, PepPI_metric[0], PepPI_metric[1],
            PepPI_metric[2], PepPI_metric[3], PepPI_metric[4], PepPI_metric[6]))
    print(
        'Evaluation - loss: {:.4f} | pep_bs_auc: {:.3f} | pep_bs_auprc: {:.3f} | pep_bs_mcc: {:.3f} | pep_bs_acc: {:.3f} | pep_bs_prec: {:.3f} | pep_bs_sen: {:.3f} | pep_bs_spec: {:.3f} | pep_bs_F1: {:.3f}'.format(
            avg_loss, pep_bs_auc, pep_bs_auprc, pep_bs_mcc, pep_bs_metric[0], pep_bs_metric[1], pep_bs_metric[2],
            pep_bs_metric[3], pep_bs_metric[4]))
    print(
        'Evaluation - loss: {:.4f} | pro_bs_auc: {:.3f} | pro_bs_auprc: {:.3f} | pro_bs_mcc: {:.3f} | pro_bs_acc: {:.3f} | pro_bs_prec: {:.3f} | pro_bs_sen: {:.3f} | pro_bs_spec: {:.3f} | pro_bs_F1: {:.3f}'.format(
            avg_loss, pro_bs_auc, pro_bs_auprc, pro_bs_mcc, pro_bs_metric[0], pro_bs_metric[1], pro_bs_metric[2],
            pro_bs_metric[3], pro_bs_metric[4]))
    return avg_loss, PepPI_auc, pep_bs_auc, pro_bs_auc

def train_test(config, independent_iter):
    # 加载
    model = Network.PepGPL(config)
    model.load_state_dict(torch.load("../result/train_validation_test_result_1_1_10_shuffle_new/model_state_dict2.pth"))
    if config.cuda:
        model.cuda()
    PepPI_criterion = nn.BCELoss()

    print('=' * 50 + 'Start Testing' + '=' * 50)
    test_loss, test_PepPI_auc, test_pep_bs_auc, test_pro_bs_auc = model_eval(
        independent_iter, model, PepPI_criterion, config)
    performance = []
    print('=' * 50 + 'Train Finished' + '=' * 50)

    return model, performance


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)

    '''load configuration'''
    config = cf.get_train_config()

    '''set device'''
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    '''train procedure'''
    valid_performance = 0
    test_performance = np.array([])
    last_test_metric = 0

    test_loss_list, test_PepPI_auc_list, test_PepPI_auprc_list = [], [], []
    test_residue_auc_list, test_residue_auprc_list = [], []

    with open('../result/train_validation_test_result_1_1_10_shuffle_new/test_dataset2.pkl', 'rb') as f:
        independent_dataset = dill.load(f)
    independent_loader = DataLoader(dataset=independent_dataset, batch_size=config.batch_size, shuffle=False,
                                    drop_last=True)
    # with open('../result/test_dataloader_save_1.pkl', 'rb') as f:
    #     independent_loader = dill.load(f)
    model, best_performance = train_test(config, independent_loader)
    test_performance = np.append(test_performance, best_performance)

    # test_performance = test_performance.reshape(-1, 5)
    # for result in test_performance:
    #     print(
    #         'k-fold - PepPI_auc: {:.4f} | PepPI_auprc: {:.4f} | pep_bs_auc: {:.4f} | pep_bs_auprc: {:.4f} | pep_bs_mcc: {:.4f}'.format(
    #             result[0], result[1], result[2], result[3], result[4]))
    # finaly_result = np.average(test_performance, axis=0)
    # print('=' * 50 + 'final-result' + '=' * 50)
    # print(
    #     'final-result - PepPI_auc: {:.4f} | PepPI_auprc: {:.4f} | pep_bs_auc: {:.4f} | pep_bs_auprc: {:.4f} | pep_bs_mcc: {:.4f}'.format(
    #         finaly_result[0], finaly_result[1], finaly_result[2], finaly_result[3], finaly_result[4]))

