import pandas as pd
import numpy as np
import re
import pickle
import csv
import time
import torch

# [PAD]:0 [UNK]:1 [CLS]:2 [SEP]:3 [MASK]:4
amino_acid_set = {"L": 5, "A": 6, "G": 7, "V": 8, "E": 9, "S": 10, "I": 11, "K": 12, "R": 13, "D": 14, "T": 15,
                  "P": 16, "N": 17, "Q": 18, "F": 19, "Y": 20, "M": 21, "H": 22, "C": 23, "W": 24, "X": 25}


def binding_vec_pos(bs_str, N):
    # if bs_str == 'NoBinding':
    #     print('Error! This record is positive.')
    #     return None
    if bs_str == '-99999':
        bs_vec = np.zeros(N)
        # bs_vec.fill(-99999)
        return bs_vec
    else:
        bs_list = [int(x) for x in bs_str.split(',')]
        bs_list = [x for x in bs_list if x < N]
        bs_vec = np.zeros(N)
        bs_vec[bs_list] = 1

        return bs_vec


def binding_vec_neg(bs_str, N):
    # if bs_str != 'NoBinding':
    #     print('Error! This record is negative.')
    #     return None
    # else:
    bs_vec = np.zeros(N)
    return bs_vec


def load_seq_label(dataset_path, pad_pep_len, pad_pro_len):
    interaction_df = pd.read_csv(dataset_path, sep='\t', header=0,
                                 keep_default_na=False, na_values=[''])
    interaction_df = interaction_df.reset_index(drop=True)
    prot_seq_df = pd.read_csv('../preprocessing/prot_seq_index.csv', sep='\t', header=0, keep_default_na=False,
                              na_values=[''])
    pep_seq_df = pd.read_csv('../preprocessing/pep_seq_index.csv', sep='\t', header=0, keep_default_na=False,
                             na_values=[''])
    prot_seq_df = prot_seq_df.reset_index(drop=True)
    pep_seq_df = pep_seq_df.reset_index(drop=True)
    with open('../preprocessing/protein_feature_dict', 'rb') as f:
        protein_feature_dict = pickle.load(f)
    with open('../preprocessing/peptide_feature_dict', 'rb') as f:
        peptide_feature_dict = pickle.load(f)
    with open('../preprocessing/protein_mask_dict', 'rb') as f:
        protein_mask_dict = pickle.load(f)
    with open('../preprocessing/peptide_mask_dict', 'rb') as f:
        peptide_mask_dict = pickle.load(f)
    with open('../preprocessing/protein_ss_dict', 'rb') as f:
        protein_ss3_dict = pickle.load(f)
    with open('../preprocessing/peptide_ss_dict', 'rb') as f:
        peptide_ss3_dict = pickle.load(f)
    with open('../preprocessing/protein_dense_dict', 'rb') as f:
        protein_dense_dict = pickle.load(f)
    with open('../preprocessing/peptide_dense_dict', 'rb') as f:
        peptide_dense_dict = pickle.load(f)
    with open('../preprocessing/protein_2_feature_dict', 'rb') as f:
        protein_2_feature_dict = pickle.load(f)
    with open('../preprocessing/peptide_2_feature_dict', 'rb') as f:
        peptide_2_feature_dict = pickle.load(f)
    # with open('../preprocessing/peptide_bert_dict', 'rb') as f:
    #     peptide_bert_dict = pickle.load(f)
    # with open('../residue_differnt_preprocessing/protein_bert_dict', 'rb') as f:
    #     protein_bert_dict = pickle.load(f)

    dim1 = len(interaction_df)
    X_pep = np.empty((dim1, pad_pep_len + 2), dtype=np.long)
    X_pro = np.empty((dim1, pad_pro_len + 2), dtype=np.long)
    X_pep_mask = np.empty((dim1, pad_pep_len + 2), dtype=np.long)
    X_pro_mask = np.empty((dim1, pad_pro_len + 2), dtype=np.long)
    X_pep_ss3 = np.empty((dim1, pad_pep_len), dtype=np.long)
    X_pro_ss3 = np.empty((dim1, pad_pro_len), dtype=np.long)
    X_pep_dense = np.empty((dim1, pad_pep_len, 3))
    X_pro_dense = np.empty((dim1, pad_pro_len, 23))
    X_pep_2 = np.empty((dim1, pad_pep_len), dtype=np.long)
    X_pro_2 = np.empty((dim1, pad_pro_len), dtype=np.long)
    X_bs_flag = np.empty((dim1, 1))
    Y = np.empty((dim1, 1))
    Y_pep_bs = np.empty((dim1, pad_pep_len))
    X_prot_bs_flag = np.empty((dim1, 1))
    Y_prot_bs = np.empty((dim1, pad_pro_len))
    # X_pep_bert = torch.empty((dim1, pad_pep_len + 2, 1024))
    # X_pro_bert = torch.empty((dim1, pad_pro_len + 2, 1024))
    # num = 1
    for idx, row in interaction_df.iterrows():
        # start = time.time()
        pro_seq = row['prot_seq'].strip()
        pep_seq = row['pep_seq'].strip()
        label = int(row['label'])
        binding_idx_str = row['binding_idx'].strip()
        prot_flag = 0
        prot_idx_str = row['prot_binding_idx']
        flag = 0
        pep_bs_vec = np.empty(pad_pep_len)
        prot_bs_vec = np.empty(pad_pro_len)
        if label == 1:
            pep_bs_vec = binding_vec_pos(binding_idx_str, pad_pep_len)
            if binding_idx_str == '-99999':
                flag = 0.0
            else:
                flag = 1.0
            if prot_idx_str == '-99999':
                prot_bs_vec = binding_vec_neg(prot_idx_str, pad_pro_len)
                prot_flag = 0.0
            else:
                bs_list = [num for num, x in enumerate(prot_idx_str) if x == '1' and num < pad_pro_len]
                prot_bs_vec = np.zeros(pad_pro_len)
                prot_bs_vec[bs_list] = 1
                prot_flag = 1.0
        elif label == 0:
            flag = 0.0
            pep_bs_vec = binding_vec_neg(binding_idx_str, pad_pep_len)
            prot_bs_vec = binding_vec_neg(prot_idx_str, pad_pro_len)
        prot_idx = prot_seq_df[prot_seq_df['prot_seq'] == pro_seq].index.tolist()[0]
        pep_idx = pep_seq_df[pep_seq_df['pep_seq'] == pep_seq].index.tolist()[0]

        X_pep_mask[idx, :] = peptide_mask_dict[pep_idx]
        X_pro_mask[idx, :] = protein_mask_dict[prot_idx]
        X_pep[idx, :] = peptide_feature_dict[pep_idx]
        X_pro[idx, :] = protein_feature_dict[prot_idx]
        X_pep_ss3[idx, :] = peptide_ss3_dict[pep_idx]
        X_pro_ss3[idx, :] = protein_ss3_dict[prot_idx]
        X_pep_dense[idx, :, :] = peptide_dense_dict[pep_idx]
        X_pro_dense[idx, :, :] = protein_dense_dict[prot_idx]
        X_pep_2[idx, :] = peptide_2_feature_dict[pep_idx]
        X_pro_2[idx, :] = protein_2_feature_dict[prot_idx]
        X_bs_flag[idx, :] = np.array([flag])
        Y[idx, :] = np.array([label])
        Y_pep_bs[idx, :] = pep_bs_vec
        X_prot_bs_flag[idx, :] = np.array([prot_flag])
        Y_prot_bs[idx, :] = prot_bs_vec
        # X_pep_bert[idx, :] = peptide_bert_dict[pep_idx]
        # X_pro_bert[idx, :] = protein_bert_dict[prot_idx]
        if idx % 1000 == 0:
            print('finish task number:', idx)

    return X_pep, X_pro, X_pep_mask, X_pro_mask, X_pep_ss3, X_pro_ss3, X_pep_dense, X_pro_dense, X_pep_2, X_pro_2, X_bs_flag, Y, Y_pep_bs, X_prot_bs_flag, Y_prot_bs


def load_data(config):
    X_pep, X_pro, X_pep_mask, X_pro_mask, X_pep_ss3, X_pro_ss3, X_pep_dense, X_pro_dense, X_pep_2, X_pro_2, X_bs_flag, Y, Y_pep_bs, X_prot_bs_flag, Y_prot_bs = load_seq_label(
        config.path_dataset, config.pad_pep_len, config.pad_pro_len)

    return X_pep, X_pro, X_pep_mask, X_pro_mask, X_pep_ss3, X_pro_ss3, X_pep_dense, X_pro_dense, X_pep_2, X_pro_2, X_bs_flag, Y, Y_pep_bs, X_prot_bs_flag, Y_prot_bs
