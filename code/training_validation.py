import os
import pickle
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from utils import load_data, rmse, mse, pearson, spearman, ci, roc_auc, pr_auc


def kfold_validation(task: str, dataset: str, setting: str, preset=None, ex_model=[]) -> None:
    """
    Perform k-fold validation for the given task, dataset, and setting.
    """
    assert task in ["dti", "dta", "moa"], "task should be in ('dti', 'dta', 'moa')."
    if task == "dti":
        assert dataset in [
            "yamanishi_08",
            "hetionet",
        ], f"dataset should be in ('yamanishi_08', 'hetionet') for {task} task."
        dataset_path = "../data/dti/" + dataset + "/"
        k_folds = 10
        eval_metric = "roc_auc"
        res_all = pd.DataFrame(columns=["AUROC", "AUPR"])

    elif task == "dta":
        assert dataset in [
            "davis",
            "kiba",
        ], f"dataset should be in ('davis','kiba') for {task} task."
        dataset_path = "../data/dta/" + dataset + "/"
        k_folds = 5
        eval_metric = None
        res_all = pd.DataFrame(columns=["RMSE", "MSE", "Pearson", "Spearman", "CI"])

    else:
        assert dataset in [
            "activation",
            "inhibition",
        ], f"dataset should be in ('activation', 'inhibition') for {task} task."
        dataset_path = "../data/moa/" + dataset + "/"
        k_folds = 5
        eval_metric = "roc_auc"
        res_all = pd.DataFrame(columns=["AUROC", "AUPR"])

    assert (
        setting in ["warm_start", "drug_coldstart", "protein_coldstart"]
    ), "validation setting should be in ('warm_start', 'drug_coldstart', 'protein_coldstart')."

    comp_feat = pickle.load(open(dataset_path + "features/compound_features.pkl", "rb"))
    prot_feat = pickle.load(open(dataset_path + "features/protein_features.pkl", "rb"))
    folds_path = dataset_path + "data_folds/" + setting + "/"

    print(f"Evaluating the model on {dataset} dataset under {setting} setting ...")
    for i in range(k_folds):
        print("fold:", i + 1)
        train_data, test_data = load_data(folds_path, i, comp_feat, prot_feat)
        test_data_nolab = test_data.drop(columns=["y"])
        predictor = TabularPredictor(label="y", eval_metric=eval_metric).fit(
            train_data=train_data, excluded_model_types=ex_model, presets=preset
        )

        if task == "dta":
            pred_scores = predictor.predict(test_data_nolab)

            G, P = np.array(test_data["y"]), np.array(pred_scores)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]

            print(
                f"fold: {i+1}, RMSE: {ret[0]}, MSE: {ret[1]}, Pearson: {ret[2]}, Spearman: {ret[3]}, CI: {ret[4]}"
            )
            res_all.loc[i] = ret

        else:
            pred_probs = predictor.predict_proba(test_data_nolab)

            auroc = roc_auc(np.array(test_data["y"]), np.array(pred_probs.iloc[:, 1]))
            aupr = pr_auc(np.array(test_data["y"]), np.array(pred_probs.iloc[:, 1]))

            print(f"AUROC and AUPR of {i+1}-fold is {auroc}, and {aupr}")
            res_all.loc[i] = [auroc, aupr]

    res_mean = res_all.mean(axis=0)
    print("all results")
    print(res_all)
    print("mean of results")
    print(res_mean)

    os.makedirs("../results/", exist_ok=True)
    res_all.to_csv(
        "../results/" + dataset + "_" + setting + ".csv", index=None, sep="\t"
    )


if __name__ == "__main__":
    task, dataset, setting = sys.argv[1], sys.argv[2], sys.argv[3]
    kfold_validation(task, dataset, setting)
