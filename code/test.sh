#!/bin/bash

python training_validation.py dti yamanishi_08 warm_start
python training_validation.py dta davis warm_start
python training_validation.py moa activation warm_start
