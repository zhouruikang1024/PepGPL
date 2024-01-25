from collections import defaultdict
import pandas as pd
import numpy as np
import re

import torch

from configuration import config as cf
import pickle
import math
from transformers import BertModel, BertTokenizer

# [PAD]:0 [UNK]:1 [CLS]:2 [SEP]:3 [MASK]:4
amino_acid_set = {"L": 5, "A": 6, "G": 7, "V": 8, "E": 9, "S": 10, "I": 11, "K": 12, "R": 13, "D": 14, "T": 15,
                  "P": 16, "N": 17, "Q": 18, "F": 19, "Y": 20, "M": 21, "H": 22, "C": 23, "W": 24, "X": 25}

amino_acid_num = 21

ss_set = {"H": 1, "C": 2, "E": 3}
ss_number = 3
residue_list = list(amino_acid_set.keys())
ss_list = list(ss_set.keys())
new_key_list = []
for i in residue_list:
    for j in ss_list:
        str_1 = str(i) + str(j)
        new_key_list.append(str_1)

new_value_list = [x + 1 for x in list(range(amino_acid_num * ss_number))]

seq_ss_set = dict(zip(new_key_list, new_value_list))
seq_ss_number = amino_acid_num * ss_number  # 63

physicochemical_set = {'A': 1, 'C': 3, 'B': 7, 'E': 5, 'D': 5, 'G': 2, 'F': 1,
                       'I': 1, 'H': 6, 'K': 6, 'M': 1, 'L': 1, 'O': 7, 'N': 4,
                       'Q': 4, 'P': 1, 'S': 4, 'R': 6, 'U': 7, 'T': 4, 'W': 2,
                       'V': 1, 'Y': 4, 'X': 7, 'Z': 7}


def get_seq_dict(prot_dir):
    seq_dict = {}
    with open(prot_dir, 'r') as f:
        pid = ''
        for line in f.readlines():
            line = line.strip()
            if line.startswith('>'):
                pid = line.split('>')[1]
            elif line != '':
                seq_dict[pid] = line
    return seq_dict


def get_seq_id(seq, pad_seq_len):
    seq_id = np.zeros(pad_seq_len + 2)
    seq_id = seq_id.astype(np.int64)
    seq_id[0] = 2
    if len(seq) <= pad_seq_len:
        for i in range(len(seq)):
            if seq[i] in amino_acid_set:
                seq_id[i + 1] = amino_acid_set[seq[i]]
            else:
                seq_id[i] = amino_acid_set['X']
        seq_id[len(seq) + 1] = 3

    else:
        for i in range(pad_seq_len):
            if seq[i] in amino_acid_set:
                seq_id[i + 1] = amino_acid_set[seq[i]]
            else:
                seq_id[i + 1] = amino_acid_set['X']
        seq_id[pad_seq_len + 1] = 3
    return seq_id


# 要加上[cls] [sep]
def get_mask(protein_seq, pad_seq_len):
    if len(protein_seq) <= pad_seq_len:
        a = np.zeros(pad_seq_len + 2)
        a = a.astype(np.int64)
        a[:len(protein_seq) + 2] = 1
    else:
        cut_protein_seq = protein_seq[:pad_seq_len]
        a = np.zeros(pad_seq_len + 2)
        a = a.astype(np.int64)
        a[:len(cut_protein_seq) + 2] = 1
    return a


def disorder_score(short_disorder_path, long_disorder_path):
    # 提取每个氨基酸无序性评分(n*3)
    disorder_dict = defaultdict(list)
    features_dict = {}
    long_disorder = []
    short_disorder = []
    anchor_score = []
    flag = 0
    with open(short_disorder_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('>'):
                seq_id = line.split('>')[1]
                flag = 1
            elif line.startswith('#') and flag == 1:
                continue
            elif flag == 1 and line != '':
                short_disorder.append(line.split()[2])
            elif line == '' and flag == 1:
                disorder_dict[seq_id].append(short_disorder)
                flag = 0
                short_disorder = []

    with open(long_disorder_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('>'):
                seq_id = line.split('>')[1]
                flag = 1
            elif line.startswith('#') and flag == 1:
                continue
            elif flag == 1 and line != '':
                long_disorder.append(line.split()[2])
                anchor_score.append(line.split()[3])
            elif line == '' and flag == 1:
                disorder_dict[seq_id].append(long_disorder)
                disorder_dict[seq_id].append(anchor_score)
                flag = 0
                long_disorder = []
                anchor_score = []
    for key, value in disorder_dict.items():
        long = value[1]
        short = value[0]
        anchor = value[2]
        disorder_list = []
        for item in zip(long, short, anchor):
            disorder_list.append(list(item))
        features_dict[key] = disorder_list
    return features_dict


def ss_feature(sspath):
    ss_dict = {}
    with open(sspath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('>'):
                seq_id = line.split('>')[1]
            elif line != '':
                ss_dict[seq_id] = line
    return ss_dict


def label_seq_ss(ss, pad_len, res_ind):
    X = np.zeros(pad_len).astype(np.int64)
    for i, res in enumerate(ss[:pad_len]):
        X[i] = res_ind[res]
    return X


def pssm_feature(pssmpath):
    pssm = []
    with open(pssmpath) as f:
        lines = f.readlines()[3:-6]
        pssm = np.array([line.split()[2:22] for line in lines], dtype=int)
    return pssm


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_array = np.vectorize(sigmoid)


def padding_sigmoid_pssm(x, N):
    x = sigmoid_array(x)
    padding_array = np.zeros([N, x.shape[1]])
    if x.shape[0] >= N:  # sequence is longer than N
        padding_array[:N, :x.shape[1]] = x[:N, :]
    else:
        padding_array[:x.shape[0], :x.shape[1]] = x
    return padding_array


def padding_intrinsic_disorder(x, N):
    x = np.array(x)
    padding_array = np.zeros([N, x.shape[1]])  # ！！！这里保存一下，刚多了个s字母
    if x.shape[0] >= N:  # sequence is longer than N
        padding_array[:N, :x.shape[1]] = x[:N, :]
    else:
        padding_array[:x.shape[0], :x.shape[1]] = x
    return padding_array


if __name__ == '__main__':
    # # step0: generate dataset peptide and protein index
    pep_seq_dict = get_seq_dict('../all_seq_data/all_pep_seq_no_pad.fa')
    prot_seq_dict = get_seq_dict('../all_seq_data/all_prot_seq.fa')
    pep_seq_df = pd.DataFrame.from_dict(pep_seq_dict, orient='index', columns=['pep_seq'])
    pep_seq_df = pep_seq_df.reset_index().rename(columns={'index': 'pid'})
    prot_seq_df = pd.DataFrame.from_dict(prot_seq_dict, orient='index', columns=['prot_seq'])
    prot_seq_df = prot_seq_df.reset_index().rename(columns={'index': 'pid'})
    pep_seq_df.to_csv('../preprocessing/pep_feature_index.csv', encoding='utf-8', index=False, sep='\t')
    prot_seq_df.to_csv('../preprocessing/prot_feature_index.csv', encoding='utf-8', index=False, sep='\t')

    # step1: generate all sequence feature
    # config = cf.get_train_config()
    # pad_prot_len = config.pad_pro_len
    # pad_pep_len = config.pad_pep_len
    # # prot_bert = BertModel.from_pretrained("../prot_bert_bfd")
    # peptide_bert_dict = {}
    # protein_bert_dict = {}
    # peptide_feature_dict = {}
    # protein_feature_dict = {}
    # peptide_mask_dict = {}
    # protein_mask_dict = {}
    # peptide_ss_dict = {}
    # protein_ss_dict = {}
    # peptide_2_feature_dict = {}
    # protein_2_feature_dict = {}
    # peptide_dense_feature_dict = {}
    # protein_dense_feature_dict = {}
    # pep_index_df = pd.read_csv('../preprocessing/pep_feature_index.csv', sep='\t', header=0,
    #                            keep_default_na=False, na_values=[''])
    # prot_index_df = pd.read_csv('../preprocessing/prot_feature_index.csv', sep='\t', header=0,
    #                             keep_default_na=False, na_values=[''])
    # pep_ss_dict = ss_feature('../all_seq_data/all_pep_seq.fa.ss')
    # prot_ss_dict = ss_feature('../all_seq_data/all_prot_seq.out.ss')
    # pep_disorder_dict = disorder_score('../all_seq_data/all_pep_seq_no_pad_short_disorder.result',
    #                                    '../all_seq_data/all_pep_seq_no_pad_long_disorder.result')
    # pro_disorder_dict = disorder_score('../all_seq_data/all_prot_seq_short_disorder.result',
    #                                     '../all_seq_data/all_prot_seq_long_disorder.result')
    #
    # interaction_df = pd.read_csv(config.path_dataset, sep='\t', header=0, keep_default_na=False, na_values=[''])
    # prot_seq_df = interaction_df.loc[:, ['prot_seq']]
    # prot_seq_df = prot_seq_df.drop_duplicates()
    # print(len(prot_seq_df))  # 4130
    # prot_seq_df = prot_seq_df.reset_index(drop=True)
    # prot_seq_df.to_csv('../preprocessing/prot_seq_index.csv', encoding='utf-8', index=False, sep='\t')
    # for idx, row in prot_seq_df.iterrows():
    #     prot_seq = row['prot_seq']
    #     df1 = prot_index_df.loc[prot_index_df.prot_seq == prot_seq]
    #     if len(df1) > 1:
    #         print('have duplicate sequence')
    #         break
    #     else:
    #         index = df1.iloc[0, 0]
    #
    #     prot_encoded_input = {}
    #     prot_seq = re.sub(r"[UZOB]", "X", prot_seq)
    #     prot_encode = get_seq_id(prot_seq, pad_prot_len)
    #     prot_mask = get_mask(prot_seq, pad_prot_len)
    #
    #     prot_input_ids = torch.tensor(prot_encode, dtype=torch.int64).unsqueeze(0)
    #     prot_attention_mask = torch.tensor(prot_mask, dtype=torch.int64).unsqueeze(0)
    #     prot_encoded_input['input_ids'] = prot_input_ids
    #     prot_encoded_input['token_type_ids'] = torch.zeros([1, pad_prot_len + 2], dtype=torch.int64)
    #     prot_encoded_input['attention_mask'] = prot_attention_mask
    #     # with torch.no_grad():
    #     #     prot_bert_encode = prot_bert(**prot_encoded_input)[0]
    #     # protein_bert_dict[idx] = prot_bert_encode
    #
    #     prot_ss3 = prot_ss_dict[str(index)]
    #     if len(prot_ss3) != len(prot_seq):
    #         print('prot idx error!', index)
    #     aa_ss = [''.join(i) for i in zip(prot_seq, prot_ss3)]
    #     prot_aa_ss = label_seq_ss(aa_ss, pad_prot_len, seq_ss_set)
    #     prot_2_feature = label_seq_ss(prot_seq, pad_prot_len, physicochemical_set)
    #     prot_disorder = padding_intrinsic_disorder(pro_disorder_dict[str(index)], pad_prot_len)
    #     prot_pssm = pssm_feature('../all_seq_data/protein_pssm/prot' + str(index) + '.pssm')
    #     prot_pssm = padding_sigmoid_pssm(prot_pssm, pad_prot_len)
    #
    #     protein_feature_dict[idx] = prot_encode
    #     protein_mask_dict[idx] = prot_mask
    #     protein_ss_dict[idx] = prot_aa_ss
    #     protein_dense_feature_dict[idx] = np.concatenate((prot_pssm, prot_disorder), axis=1)
    #     protein_2_feature_dict[idx] = prot_2_feature
    #     if idx % 100 == 0:
    #         print('finish protein task：', idx)
    #
    # pep_seq_df = interaction_df.loc[:, ['pep_seq']]
    # pep_seq_df = pep_seq_df.drop_duplicates()
    # print(len(pep_seq_df))  # 6758
    # pep_seq_df = pep_seq_df.reset_index(drop=True)
    # pep_seq_df.to_csv('../preprocessing/pep_seq_index.csv', encoding='utf-8', index=False, sep='\t')
    # for idx, row in pep_seq_df.iterrows():
    #     pep_seq = row['pep_seq']
    #     df1 = pep_index_df.loc[pep_index_df.pep_seq == pep_seq]
    #     if len(df1) > 1:
    #         print('have duplicate sequence')
    #         break
    #     else:
    #         index = df1.iloc[0, 0]
    #
    #     pep_seq = re.sub(r"[UZOB]", "X", pep_seq)
    #     pep_encode = get_seq_id(pep_seq, pad_pep_len)
    #     pep_mask = get_mask(pep_seq, pad_pep_len)
    #
    #     # pep_encoded_input = {}
    #     # pep_input_ids = torch.tensor(pep_encode, dtype=torch.int64).unsqueeze(0)
    #     # pep_attention_mask = torch.tensor(pep_mask, dtype=torch.int64).unsqueeze(0)
    #     # pep_encoded_input['input_ids'] = pep_input_ids
    #     # pep_encoded_input['token_type_ids'] = torch.zeros([1, pad_pep_len + 2], dtype=torch.int64)
    #     # pep_encoded_input['attention_mask'] = pep_attention_mask
    #     # with torch.no_grad():
    #     #     pep_bert_encode = prot_bert(**pep_encoded_input)[0]
    #     # peptide_bert_dict[idx] = pep_bert_encode
    #
    #     pep_ss3 = pep_ss_dict[str(index)][:len(pep_seq)]  # 肽长度都填充到30了，所以要取原始序列
    #     aa_ss = [''.join(i) for i in zip(pep_seq, pep_ss3)]
    #     pep_aa_ss = label_seq_ss(aa_ss, pad_pep_len, seq_ss_set)
    #     pep_2_feature = label_seq_ss(pep_seq, pad_pep_len, physicochemical_set)
    #     pep_disorder = padding_intrinsic_disorder(pep_disorder_dict[str(index)], pad_pep_len)
    #
    #     peptide_feature_dict[idx] = pep_encode
    #     peptide_mask_dict[idx] = pep_mask
    #     peptide_ss_dict[idx] = pep_aa_ss
    #     peptide_dense_feature_dict[idx] = pep_disorder
    #     peptide_2_feature_dict[idx] = pep_2_feature
    #     if idx % 100 == 0:
    #         print('finish peptide task：', idx)
    #
    # with open('../preprocessing/peptide_feature_dict', 'wb') as f:
    #     pickle.dump(peptide_feature_dict, f)
    # with open('../preprocessing/protein_feature_dict', 'wb') as f:
    #     pickle.dump(protein_feature_dict, f)
    # with open('../preprocessing/peptide_mask_dict', 'wb') as f:
    #     pickle.dump(peptide_mask_dict, f)
    # with open('../preprocessing/protein_mask_dict', 'wb') as f:
    #     pickle.dump(protein_mask_dict, f)
    # with open('../preprocessing/peptide_ss_dict', 'wb') as f:
    #     pickle.dump(peptide_ss_dict, f)
    # with open('../preprocessing/protein_ss_dict', 'wb') as f:
    #     pickle.dump(protein_ss_dict, f)
    # with open('../preprocessing/peptide_dense_dict', 'wb') as f:
    #     pickle.dump(peptide_dense_feature_dict, f)
    # with open('../preprocessing/protein_dense_dict', 'wb') as f:
    #     pickle.dump(protein_dense_feature_dict, f)
    # with open('../preprocessing/peptide_2_feature_dict', 'wb') as f:
    #     pickle.dump(peptide_2_feature_dict, f)
    # with open('../preprocessing/protein_2_feature_dict', 'wb') as f:
    #     pickle.dump(protein_2_feature_dict, f)
    #
