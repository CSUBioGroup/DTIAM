import os
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.model_selection import KFold


def split(dataset, n_splits=10, ratio=10):
    
    path = '../data/dti/' + dataset + '/'
    dti = pd.read_csv(path + 'dti.csv', sep='\t', header=None)
    drugs = list(dti[0].drop_duplicates())
    targets = list(dti[2].drop_duplicates())

    dti_all = list(itertools.product(drugs, targets))
    dti_pos = pd.read_csv(path + 'dti.csv', sep='\t', header=None)
    dti_pos = [(row[0], row[2]) for _, row in dti_pos.iterrows()]
    dti_neg_all = list(set(dti_all) - set(dti_pos))
    assert len(dti_all) == len(dti_neg_all) + len(dti_pos)

    dti_neg = random.sample(dti_neg_all, len(dti_pos) * ratio)
    dti_pos = [tup + (1,) for tup in dti_pos]
    dti_neg = [tup + (0,) for tup in dti_neg]
    dti = np.array(dti_pos + dti_neg)

    drugs_arr = np.array(drugs)
    targets_arr = np.array(targets)

    def save_fold(data, idx, name, setting):
        data = pd.DataFrame(data, columns=['DrugID', 'TargetID', 'label'])
        fold_path = path + 'data_folds/' + setting
        os.makedirs(fold_path, exist_ok=True)
        data.to_csv(fold_path + '/' + name + '_fold_' + str(idx) + '.csv', index=None)

    def split_warm(setting='warm_start'):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        idx = 0
        for train_index, test_index in kf.split(dti):
            dti_train = dti[train_index]
            dti_test = dti[test_index]
            save_fold(dti_train, idx, 'train', setting)
            save_fold(dti_test, idx, 'test', setting)
            idx += 1

    def split_drug_cold(setting='drug_coldstart'):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        idx = 0
        for train_index, test_index in kf.split(drugs_arr):
            dti_train = dti[np.isin(dti[:, 0], drugs_arr[train_index])]
            dti_test = dti[np.isin(dti[:, 0], drugs_arr[test_index])]
            save_fold(dti_train, idx, 'train', setting)
            save_fold(dti_test, idx, 'test', setting)
            idx += 1

    def split_protein_cold(setting='protein_coldstart'):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        idx = 0
        for train_index, test_index in kf.split(targets_arr):
            dti_train = dti[np.isin(dti[:, 1], targets_arr[train_index])]
            dti_test = dti[np.isin(dti[:, 1], targets_arr[test_index])]
            save_fold(dti_train, idx, 'train', setting)
            save_fold(dti_test, idx, 'test', setting)
            idx += 1
    
    split_warm()
    split_drug_cold()
    split_protein_cold()
    
    print(dataset + ' dataset split completed.')


if __name__ == '__main__':
    
    for dataset in ["yamanishi_08", "hetionet"]:
        split(dataset)
