import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MolDataset(Dataset):
    def __init__(self, corpus_path, vocab, on_memory=True):
        super(Dataset).__init__()
        
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.on_memory = on_memory
        
        with open(corpus_path, 'r') as f:
            if on_memory:
                self.data = [line[:-1].split('\t') for line in tqdm.tqdm(f, desc="Loading Dataset")]
                self.len = len(self.data)
            else:
                self.len = 0
                for _ in tqdm.tqdm(f, desc="Loading Dataset"):
                    self.len += 1
        
        if not on_memory:
            self.file = open(corpus_path, 'r')
    
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.on_memory:
            sentence, fgs, desc = self.data[item]
        else:
            try:
                line = self.file.__next__()
            except:
                self.file.close()
                self.file = open(self.corpus_path, 'r')
                line = self.file.__next__()
            
            sentence, fgs, desc = line[:-1].split('\t')

        token_ids, mask_labels = self.replace_mask_tokens(sentence)
        token_ids = [self.vocab.cls_index] + token_ids + [self.vocab.sep_index]
        mask_labels = [self.vocab.pad_index] + mask_labels + [self.vocab.pad_index]

        fg_labels = self.prepare_motif(fgs)
        desc_labels = desc.split()

        output = {
            "token_ids": np.array(token_ids),
            "mask_labels": np.array(mask_labels),
            "fg_labels": np.array(fg_labels),
            "desc_labels": np.array(desc_labels, dtype=np.float32)
        }
        return output

    def replace_mask_tokens(self, sentence):
        tokens = sentence.split()
        labels = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                labels.append(self.vocab.pad_index)

        return tokens, labels
    
    def prepare_motif(self, fgs):
        fgs = fgs.split()
        fg_labels = [0] * len(self.vocab.fgtoi)
        for fg in fgs:
            if fg in self.vocab.fgtoi:
                fg_labels[self.vocab.fgtoi[fg]] = 1
        return fg_labels


def collate_batch(batch):
    maxn = max([item["token_ids"].shape[0] for item in batch])
    
    token_ids = np.zeros((len(batch), maxn))
    attention_mask = np.zeros((len(batch), maxn, maxn)) - 10000.0
    mask_labels = np.zeros((len(batch), maxn))
    fg_labels = np.zeros((len(batch), len(batch[0]["fg_labels"])))
    desc_labels = np.zeros((len(batch), len(batch[0]["desc_labels"])))
    
    for i, item in enumerate(batch):
        n = item["token_ids"].shape[0]

        token_ids[i, :n] = item["token_ids"]
        attention_mask[i, :, :n] = 0
        mask_labels[i, :n] = item["mask_labels"]
        fg_labels[i, :] = item["fg_labels"]
        desc_labels[i, :] = item["desc_labels"]
    
    output = {
        "token_ids": torch.LongTensor(token_ids), 
        "attention_mask": torch.FloatTensor(attention_mask), 
        "mask_task": torch.LongTensor(mask_labels),
        "motif_task": torch.LongTensor(fg_labels), 
        "desc_task": torch.FloatTensor(desc_labels)
    }
    
    return output


def mol_dataloader(corpus_path, vocab, batch_size=64, shuffle=False, num_workers=8, on_memory=True):
    dataset = MolDataset(corpus_path, vocab, on_memory)
    if not on_memory:
        num_workers = 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_batch)
