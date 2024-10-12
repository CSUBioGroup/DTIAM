from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def load_data(data_path, fold_idx, comp_feat, prot_feat):
    """Load training and testing data."""
    print("Loading data ...")
    train = pd.read_csv(data_path + "train_fold_" + str(fold_idx) + ".csv")
    test = pd.read_csv(data_path + "test_fold_" + str(fold_idx) + ".csv")
    train.columns = ["cid", "pid", "label"]
    test.columns = ["cid", "pid", "label"]
    return pack(train, comp_feat, prot_feat), pack(test, comp_feat, prot_feat)


def pack(data, comp_feat, prot_feat):
    """Pack compound and protein features into a dataframe."""
    vecs = []
    for i in range(len(data)):
        cid, pid = data.iloc[i, :2]
        vecs.append(list(comp_feat[str(cid)]) + list(prot_feat[pid]))
    vecs_df = pd.DataFrame(vecs)
    vecs_df["y"] = data["label"]
    return vecs_df


def roc_auc(y, pred):
    """Compute the ROC AUC score."""
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def pr_auc(y, pred):
    """Compute the Precision-Recall AUC score."""
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def rmse(y, f):
    """Compute the Root Mean Squared Error."""
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    """Compute the Mean Squared Error."""
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    """Compute the Pearson correlation coefficient."""
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    """Compute the Spearman correlation coefficient."""
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    """Compute the Concordance Index."""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci
