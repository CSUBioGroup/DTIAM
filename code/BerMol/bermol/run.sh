python trainer.py \
--train_corpus_path  ../data/pubchem1K_train.corpus \
--val_corpus_path ../data/pubchem1K_val.corpus \
--vocab_path ../data/pubchem1K.vocab \
--bermol_path ../checkpoints/bermol_pubchem10M_0725/ \
--epochs 100 \
--device cuda:0 \
