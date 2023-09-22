#!/bin/bash

python data_process/data_split_dti.py
python data_process/data_split_dta.py
python data_process/data_split_moa.py
python data_process/extract_feature.py
