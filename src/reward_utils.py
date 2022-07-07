import numpy as np
import pandas as pd
import networkx as nx
import scipy.signal
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from .sascorer import calculateScore


def canonicalize_smile(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def external_diversity(smi_new, smi_bank):
    td = 0
    mol_new = Chem.MolFromSmiles(smi_new)
    fps_new = AllChem.GetMorganFingerprint(mol_new, 6)
    fps_B = []
    for smi in smi_bank:
        try:
            smi = canonicalize_smile(smi)
            mol = Chem.MolFromSmiles(smi)
            fps_B.append(AllChem.GetMorganFingerprint(mol, 6))
        except:
            print('ERROR: Invalid SMILES!')

    for i in range(len(fps_B)):
        ts = 1 - DataStructs.TanimotoSimilarity(fps_new, fps_B[i])
        td += ts

    td = td / len(fps_B)
    return td


def get_cycle_reward(smi):
    smi = canonicalize_smile(smi)
    mol = Chem.MolFromSmiles(smi)
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def get_logp_reward(smi):
    smi = canonicalize_smile(smi)
    mol = Chem.MolFromSmiles(smi)
    return MolLogP(mol)


def get_sa_reward(smi):
    try:
        smi = canonicalize_smile(smi)
        mol = Chem.MolFromSmiles(smi)
    except:
        return 0
    return calculateScore(mol)


def get_qed_reward(smi):
    try:
        smi = canonicalize_smile(smi)
        mol = Chem.MolFromSmiles(smi)
    except:
        return 0
    return qed(mol)


def _canonicalize_smiles(smi):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return np.nan
    return smi


def _parallel_get_fps(smi):
    smi = _canonicalize_smiles(smi)
    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def _parallel_get_int_dists(fp_new, fps):
    dist = DataStructs.BulkTanimotoSimilarity(fp_new, fps)
    return np.sum(dist)


def _parallel_get_ext_dists(fp_new, fps):
    sm = DataStructs.BulkTanimotoSimilarity(fp_new, fps)
    return np.max(sm), np.argmax(sm)


def get_sim_reward(smi, ref_list):
    fp = _parallel_get_fps(smi)
    ref_fps = [_parallel_get_fps(smi) for smi in ref_list]

    sim = _parallel_get_ext_dists(fp, ref_fps)
    if sim >= 0.6 and sim <= 0.8:
        return 1

    if sim < 0.6:
        return sim / 0.6

    if sim > 0.8:
        return 0.8 / sim


def get_diversity_reward(smi):
    smi_bank = pd.read_csv('data/smiles_base.csv').Smiles.values
    diversity = external_diversity(smi, smi_bank)
    if diversity > 0.9:
        return 1
    else:
        return 0


def get_penalized_logp_reward(smi):
    smi_base = pd.read_csv('data/smiles_base.csv')
    logP_mean = smi_base.logp.mean()
    logP_std = smi_base.logp.std()
    SA_mean = smi_base.sa.mean()
    SA_std = smi_base.sa.std()
    cycle_mean = smi_base.cycle.mean()
    cycle_std = smi_base.cycle.std()

    try:
        log_p = get_logp_reward(smi)
        sa = get_sa_reward(smi)
        cycle = get_cycle_reward(smi)
    except:
        # if invalid molecules are generated
        return -1

    # smaller the better
    normalized_log_p = -(log_p - logP_mean) / logP_std
    normalized_SA = (sa - SA_mean) / SA_std
    normalized_cycle = (cycle - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
