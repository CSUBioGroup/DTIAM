import pickle
import dill as pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from rdkit.Chem import Descriptors
from bermol.utils import parall_build


class MolVocab:
    def __init__(self, config: argparse.Namespace, specials: list = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]) -> None:
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4

        self.count = defaultdict(int)
        self.stoi = defaultdict(lambda: len(self.stoi))
        self.fgtoi = defaultdict(lambda: len(self.fgtoi))
        [self.stoi[s] for s in specials]
        self.descriptor = sorted([x[0] for x in Descriptors._descList])
        self.max_len = config.max_len
        self.min_freq = config.min_freq

    def build_corpus(self, mol_path: str, corpus_path: str = None, update: bool = True) -> "MolVocab":
        print("Building Vocab and Corpus...")
        if corpus_path is None:
            corpus_path = mol_path + ".corpus"

        f_mol = open(mol_path, "r")
        corpus_res = Parallel(n_jobs=-1)(
            delayed(parall_build)(line.strip(), self.max_len) for line in tqdm(f_mol)
        )
        f_mol.close()

        cnt = 0
        print("Save Corpus...")
        with open(corpus_path, "w") as f_cor:
            for data in tqdm(corpus_res):
                if not data:
                    continue

                sentence, fgs, desc = data
                if len(sentence) > self.max_len:
                    print(sentence)
                f_cor.write(
                    "\t".join([" ".join(sentence), " ".join(fgs), " ".join(desc)])
                    + "\n"
                )
                cnt += 1

                if update:
                    for s in sentence:
                        self.count[s] += 1
                    for fg in fgs:
                        self.fgtoi[fg]
        if update:
            for s in self.count:
                if self.count[s] >= self.min_freq:
                    self.stoi[s]

        print(
            "Building Complete, Number of Sample: %s, Vocab Size: %s, Number of Functinal Grounp: %s, Number of Molecular Descriptor: %s"
            % (cnt, len(self), len(self.fgtoi), len(self.descriptor))
        )
        return self

    def __len__(self) -> int:
        return len(self.stoi)

    def save_vocab(self, vocab_path: str, silent: bool = False) -> None:
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
        if not silent:
            print(
                f'Vocab saved. To load, use: vocab = MolVocab.load_vocab("{vocab_path}")'
            )

    @staticmethod
    def load_vocab(vocab_path: str) -> "MolVocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_mol_path",
        type=str,
        required=True,
        help="input molecular path of training set",
    )
    parser.add_argument(
        "--val_mol_path",
        type=str,
        default=None,
        help="input molecular path of validation set",
    )
    parser.add_argument(
        "--train_corpus_path",
        type=str,
        required=True,
        help="output molecular corpus path of training set",
    )
    parser.add_argument(
        "--val_corpus_path",
        type=str,
        default=None,
        help="output molecular corpus path of validation set",
    )
    parser.add_argument(
        "--vocab_path", type=str, required=True, help="output molecular vocab path"
    )
    parser.add_argument(
        "--max_len", type=int, default=128, help="max length of molecular sentence"
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="min frequency of the molecular substructure",
    )

    config = parser.parse_args()

    vocab = MolVocab(config)
    vocab = vocab.build_corpus(config.train_mol_path, config.train_corpus_path)
    if config.val_mol_path is not None:
        vocab = vocab.build_corpus(
            config.val_mol_path, config.val_corpus_path, update=False
        )
    vocab.save_vocab(config.vocab_path)


if __name__ == "__main__":
    build()
