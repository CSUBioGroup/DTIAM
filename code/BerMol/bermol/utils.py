import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.FilterCatalog import GetFunctionalGroupHierarchy
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch.optim.lr_scheduler import LambdaLR

from bermol.model import DescPredictor, MaskPredictor, MotifPredictor

TASK_DICT = {
    "mask_task": MaskPredictor,
    "motif_task": MotifPredictor,
    "desc_task": DescPredictor,
}


def smi_to_mol(smiles: str) -> Chem.Mol:
    try:
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        smi = Chem.MolToSmiles(mol)  # standardize
        mol = Chem.MolFromSmiles(smi)
        return mol
    except:
        return None


def mol_to_sentence(mol: Chem.Mol, radius: int = 1) -> list:
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    atom_dic = {x: {r: None for r in range(radius + 1)} for x in mol_atoms}

    for element in info:
        for atom_idx, r in info[element]:
            atom_dic[atom_idx][r] = element

    sentence = []
    for atom in atom_dic:
        if atom_dic[atom][radius]:
            sentence.append(str(atom_dic[atom][radius]))
    return sentence


def get_functional_gropus(mol: Chem.Mol) -> list:
    functionalGroups = GetFunctionalGroupHierarchy()
    fgs = [
        match.filterMatch.GetName() for match in functionalGroups.GetFilterMatches(mol)
    ]
    return fgs


def get_molecular_descriptor(mol: Chem.Mol) -> list:
    descriptor = sorted([x[0] for x in Descriptors._descList])

    desc_calc = MolecularDescriptorCalculator(descriptor)
    desc = desc_calc.CalcDescriptors(mol)
    desc = np.array(desc)
    desc[~np.isfinite(desc)] = 0

    desc = (desc - distributions["min"]) / distributions["scale"]
    desc = np.clip(desc, 0, 1)
    return list(desc.astype(str))


def parall_build(smi: str, max_len: int = 64) -> tuple:
    mol = smi_to_mol(smi)
    if mol is None:
        return

    sentence = mol_to_sentence(mol)
    if not sentence:
        return

    fgs = get_functional_gropus(mol)
    desc = get_molecular_descriptor(mol)
    return sentence[:max_len], fgs, desc


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


desc_min = [
    1.290908900087874,
    -3.182750032141512,
    1.2534937959110932,
    -3.614014782299629,
    4.1645443020921835,
    -1.2081548979096097,
    13.37805378572964,
    8.785956190491824,
    0.6210635595716008,
    8.0,
    3.414213562373095,
    1.9328121551534467,
    1.9328121551534467,
    1.7320508075688772,
    0.7164060775767234,
    0.7164060775767234,
    0.33192730526045355,
    0.33192730526045355,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    54.046950192,
    0.15714285714285714,
    0.2714285714285714,
    0.4,
    0.0,
    -8.049999999999997,
    4.0,
    48.044,
    3.2451124978365313,
    3.2700000000000005,
    0.5305077970220748,
    0.0971094971373244,
    23.03489757708204,
    1.6122685185185186,
    0.012386067410748738,
    1.6122685185185186,
    -0.07008230826610218,
    0.0,
    0.0002425759502671019,
    -10.159583862333859,
    -0.8697667346936595,
    -10.19180000000002,
    12.2122,
    54.09199999999999,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    22.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -3.640462962962963,
    -4.73543150825145,
    -6.495197704081633,
    -7.026572302532124,
    -15.35712631168935,
    -52.7841722193084,
    -11.278790991322813,
    -31.835146841143864,
    -8.25510392196577,
    -22.979347338195918,
    0.006844353234188021
]

desc_scale = [
    2.012830277074508,
    1.7410386136895317,
    1.975011193847191,
    2.40369453370101,
    10.045448565146467,
    3.3557346493109894,
    113.54458295516127,
    6.828540455138452,
    7.090169453905672,
    3071.4379310492013,
    50.19990956309004,
    44.99144100492723,
    44.99144100492723,
    35.82649984049591,
    29.74713233567008,
    29.795914031598034,
    22.722099146095257,
    29.35423211151864,
    19.465792253752163,
    23.90217659714905,
    16.749124288589435,
    28.79084622014601,
    150.08046090700503,
    95.55941945186089,
    19.178148736288286,
    123.09603330285219,
    163.42641582098778,
    183.25317936555317,
    274.75939375346104,
    137.15050903737546,
    181.9910120538493,
    232.23668886400012,
    139.21127868279115,
    1407.8250254480013,
    1.842857142858143,
    2.4785714285724287,
    3.1555555555565555,
    1.000000000001,
    10.950000000000998,
    72.000000000001,
    1353.8610000000006,
    3.3174790163640584e+16,
    65.91000000000102,
    67.64949220297895,
    9507.862890502864,
    435.9210854674902,
    16.108974924120222,
    0.8573806672839107,
    16.108974924120222,
    0.8637547997360039,
    3.5435648148158148,
    0.6177749686426178,
    12.159583862334859,
    0.8573806672839107,
    28.705400000000996,
    299.2596000000009,
    1408.2930000000006,
    20.000000000001,
    26.000000000001,
    9.000000000001,
    20.000000000001,
    20.000000000001,
    8.000000000001,
    6.000000000001,
    10.000000000001,
    24.000000000001,
    15.000000000001,
    29.000000000001,
    1.000000000001,
    66.000000000001,
    8.000000000001,
    11.000000000001,
    11.000000000001,
    404.000000000001,
    104.64942420437124,
    98.16637778073844,
    71.47929462132353,
    53.16461756416455,
    45.46304612313093,
    307.12527245685584,
    59.00642419722134,
    57.11648464514997,
    83.05485279480189,
    92.8075191218611,
    270.1774237396581,
    324.26338550395207,
    140.08558815387497,
    122.47110819603962,
    30.000000000001,
    105.40280980050164,
    164.82791217563204,
    42.095132437908894,
    80.24113892108112,
    76.9327786000991,
    322.0469122998306,
    183.25317936555317,
    251.76992341480087,
    1e-12,
    90.80327725871228,
    74.43504045608964,
    74.63705581047294,
    68.99414199940786,
    141.14261939269437,
    222.24283507680838,
    85.72492701057607,
    90.64890294024708,
    322.0469122998306,
    218.3892144646189,
    54.39686199224924,
    89.77337874773002,
    1e-12,
    415.42000000000104,
    277.1206949409731,
    87.42697241821529,
    127.96571034110092,
    148.5010204851682,
    52.07406318784934,
    76.55128803922108,
    93.56809090636017,
    95.83966852853906,
    44.26160472227389,
    37.40541919660067,
    0.940810852506316
]

distributions = {"min": np.array(desc_min), "scale": np.array(desc_scale)}
