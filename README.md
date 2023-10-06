## Codes for "DTIAM: A unified framework for predicting drug-target interactions, binding affinities and activation/inhibition mechanisms"

The benchmark dataset described in this paper can be found in ./data/.

The complement of the self-supervised molecular representation learning model BerMol can be found in ./code/BerMol/. [Download](https://drive.google.com/file/d/1ZW-PQXE4FvWHx77hkUA-JsqyJUb6B-NQ/view?usp=drive_link) the pre-trained model file.

Before running the DTIAM model in ./code/, please first use ./code/data_prepare.sh to produce necessary files.

For cross validation, e.g., in moa task, using activation data and warm_start setting, run:

```python training_validation.py moa activation warm_start```

### Quick start

```bash
# Run the commandline
conda create -n dtiam python=3.7 -y
conda activate dtiam
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
cd BerMol
pip install -e .
```

### Requirements:
```
- python 3.7
- pytorch 1.12.1
- autogluon 0.5.2
- dill 0.3.4
- fair_esm 2.0.0
- joblib 1.1.0
- numpy 1.21.2
- pandas 1.3.5
- rdkit 2022.9.5
- setuptools 59.8.0
- tqdm 4.62.2
```
