import os
import pandas as pd
import numpy as np
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from sklearn.model_selection import KFold


def split(dataset, n_splits=5):
    
    fpath = '../data/dta/' + dataset + '/'
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drug_ids, prot_ids = list(ligands.keys()), list(proteins.keys())
    drugs, prots = [], []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    rows, cols = np.where(np.isnan(affinity)==False)
    dti = np.array([(rows[i], cols[i], affinity[rows[i],cols[i]]) for i in range(len(rows))])
    
    def save_fold(data, idx, name, setting):
        data = pd.DataFrame(data)
        data.columns = ['drug_idx', 'prot_idx', 'affinity']
        data['drug_id'] = data['drug_idx'].apply(lambda x:drug_ids[int(x)])
        data['protein_id'] = data['prot_idx'].apply(lambda x:prot_ids[int(x)])

        data_id = data[['drug_id', 'protein_id', 'affinity']]
        fold_path = fpath + 'data_folds/' + setting
        os.makedirs(fold_path, exist_ok=True)
        data_id.to_csv(fold_path + '/' + name + '_fold_' + str(idx) + '.csv', index=None)
    
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
        drugs_arr = np.array(range(len(drug_ids)))
        for train_index, test_index in kf.split(drugs_arr):
            dti_train = dti[np.isin(dti[:, 0], drugs_arr[train_index])]
            dti_test = dti[np.isin(dti[:, 0], drugs_arr[test_index])]
            save_fold(dti_train, idx, 'train', setting)
            save_fold(dti_test, idx, 'test', setting)
            idx += 1

    def split_protein_cold(setting='protein_coldstart'):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        idx = 0
        targets_arr = np.array(range(len(prot_ids)))
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
    
    for dataset in ['davis', 'kiba']:
        split(dataset)
