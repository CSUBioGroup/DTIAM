#!/bin/bash

train_mol_path='./data/training'
val_mol_path='./data/validation'
train_corpus_path='./data/training.corpus'
val_corpus_path='./data/validation.corpus'
vocab_path='./data/vocab'

# python bermol/vocab.py \
#     --train_mol_path $train_mol_path \
#     --val_mol_path $val_mol_path \
#     --train_corpus_path $train_corpus_path \
#     --val_corpus_path $val_corpus_path \
#     --vocab_path $vocab_path \
#     --max_len 128 \
#     --min_freq 1 \

python bermol/trainer.py \
    --train_corpus_path $train_corpus_path \
    --val_corpus_path $val_corpus_path \
    --vocab_path $vocab_path \
    --epochs 100 \
    --device 'cuda:3' \
