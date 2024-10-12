import numpy as np
import torch

from bermol.utils import mol_to_sentence, smi_to_mol


class BerMolTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def encode(self, smiles):
        mol = smi_to_mol(smiles)
        if mol is None:
            raise Exception("invalid smiles, please check the imput smiles!")

        sentence = mol_to_sentence(mol)
        token_ids = self.sentence_to_token(sentence)
        token_ids = torch.LongTensor(token_ids)
        return torch.unsqueeze(token_ids, 0)

    def sentence_to_token(self, sentence):
        token_ids = [
            self.vocab.stoi.get(token, self.vocab.unk_index) for token in sentence
        ]
        token_ids = [self.vocab.cls_index] + token_ids + [self.vocab.sep_index]
        return np.array(token_ids)
