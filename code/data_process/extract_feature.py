import os
import torch
import pickle
import dill as pickle
import json
import esm
import pandas as pd
from bermol.trainer import BerMolPreTrainer
from tqdm import tqdm
from collections import OrderedDict


def cal_comp_feat(data, model_path, device="cuda"):
    with open(model_path, "rb") as f:
        comp_model = pickle.load(f)
        comp_model.model.to(device)
        comp_model.model.eval()

    def smi_to_vec(smi):
        output = comp_model.transform(smi, device)
        return output[1].cpu().detach().numpy().reshape(-1)

    comp_feat = {}
    comp_data = data[["cid", "smi"]].drop_duplicates(subset=["cid"])
    for _, row in tqdm(comp_data.iterrows()):
        cid, smi = row[0], row[1]
        comp_feat[cid] = smi_to_vec(smi)

    return comp_feat


def cal_prot_feat(data):
    # model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    repr_layer = model.num_layers

    def seq_to_vecs(pid, seq, max_length=1022):
        data = [
            (pid, seq[:max_length]),
        ]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
        token_representations = results["representations"][repr_layer]
        sequence_representations = token_representations[0, 1:].mean(0)
        return sequence_representations.cpu().detach().numpy().reshape(-1)

    prot_feat = {}
    prot_data = data[["pid", "seq"]].drop_duplicates(subset=["pid"])
    for _, row in tqdm(prot_data.iterrows()):
        pid, seq = row[0], row[1]
        prot_feat[pid] = seq_to_vecs(pid, seq)

    return prot_feat


def extract_dti():
    for dataset in ["yamanishi_08", "hetionet"]:
        data_path = "../data/dti/" + dataset + "/"
        save_path = data_path + "features/"
        os.makedirs(save_path, exist_ok=True)

        drug_smi = pd.read_csv(data_path + "drug_smiles.csv", sep="\t")
        tar_seq = pd.read_csv(data_path + "protein_seq.csv", sep="\t")
        drug_smi.columns = ["cid", "smi"]
        tar_seq.columns = ["pid", "seq"]

        if dataset == "hetionet":
            dti = pd.read_csv(data_path + "dti.csv", sep="\t", header=None)
            drugs = list(dti[0].drop_duplicates())
            targets = list(dti[2].drop_duplicates())

            drug_smi = drug_smi[drug_smi["cid"].isin(drugs)]
            tar_seq = tar_seq[tar_seq["pid"].isin(targets)]

        print(f"Extracting compound features for {dataset} dataset ...")
        comp_feat = cal_comp_feat(drug_smi, bermol_model_path)
        with open(save_path + "compound_features.pkl", "wb") as f:
            pickle.dump(comp_feat, f)

        print(f"Extracting protein features for {dataset} dataset ...")
        prot_feat = cal_prot_feat(tar_seq)
        with open(save_path + "protein_features.pkl", "wb") as f:
            pickle.dump(prot_feat, f)


def extract_dta():
    for dataset in ["davis", "kiba"]:
        data_path = "../data/dta/" + dataset + "/"
        save_path = data_path + "features/"
        os.makedirs(save_path, exist_ok=True)

        with open(data_path + "ligands_can.txt") as f:
            ligands = json.load(f, object_pairs_hook=OrderedDict)
            drug_smi = pd.DataFrame(ligands.items(), columns=["cid", "smi"])

        with open(data_path + "proteins.txt") as f:
            proteins = json.load(f, object_pairs_hook=OrderedDict)
            tar_seq = pd.DataFrame(proteins.items(), columns=["pid", "seq"])

        print(f"Extracting compound features for {dataset} dataset ...")
        comp_feat = cal_comp_feat(drug_smi, bermol_model_path)
        with open(save_path + "compound_features.pkl", "wb") as f:
            pickle.dump(comp_feat, f)

        print(f"Extracting protein features for {dataset} dataset ...")
        prot_feat = cal_prot_feat(tar_seq)
        with open(save_path + "protein_features.pkl", "wb") as f:
            pickle.dump(prot_feat, f)


def extract_moa():
    for dataset in ["activation", "inhibition"]:
        data_path = "../data/moa/" + dataset + "/"
        save_path = data_path + "features/"
        os.makedirs(save_path, exist_ok=True)

        drug_smi = pd.read_csv(data_path + "drug_smi.csv", sep="\t")
        tar_seq = pd.read_csv(data_path + "tar_seq.csv", sep="\t")
        drug_smi.columns = ["cid", "smi"]
        tar_seq.columns = ["pid", "seq"]

        print(f"Extracting compound features for {dataset} dataset ...")
        comp_feat = cal_comp_feat(drug_smi, bermol_model_path)
        with open(save_path + "compound_features.pkl", "wb") as f:
            pickle.dump(comp_feat, f)

        print(f"Extracting protein features for {dataset} dataset ...")
        prot_feat = cal_prot_feat(tar_seq)
        with open(save_path + "protein_features.pkl", "wb") as f:
            pickle.dump(prot_feat, f)


if __name__ == "__main__":
    bermol_model_path = "../models/BerMolModel_base.pkl"
    extract_dti()
    extract_dta()
    extract_moa()
