import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from src.reward_utils import get_cycle_reward, get_logp_reward, get_qed_reward, get_sa_reward
from src.data_process_utils import standardize_smiles_error_handle
from generate_similar_mol import _canonicalize_smiles, _parallel_get_ext_dists, _parallel_get_fps


def get_file_names():
    df_rows = []
    for f in os.listdir('C://Users//yhytx//SEED//gen_samples_rl_similar_binder'):
        if f.endswith(".png"):
            smi = f.replace('.png', '')
            smi_property = get_smi_property(smi)
            df_rows.append(smi_property)
            if smi_property['similarity'] == 1:
                breakpoint()
    breakpoint()
    gen_samples_good_df = pd.DataFrame(df_rows)
    gen_samples_good_df.to_csv('generated_molecules_similar_binder.csv', index=False)


def get_smi_property(smi):
    mol = Chem.MolFromSmiles(smi)
    mol_w = MolWt(mol)
    fp = _parallel_get_fps(smi)
    sim, sim_idx = _parallel_get_ext_dists(fp, ref_fps)
    gen_sample = {}
    gen_sample["Smiles"] = smi
    gen_sample["NumAtoms"] = mol.GetNumAtoms()
    gen_sample['logp'] = np.round(get_logp_reward(smi), 4)
    gen_sample['sa'] = np.round(get_sa_reward(smi), 4)
    gen_sample['cycle'] = get_cycle_reward(smi)
    gen_sample['qed'] = np.round(get_qed_reward(smi), 4)
    gen_sample['similarity'] = sim
    gen_sample['mol_weight'] = mol_w
    gen_sample['matched'] = sim_idx + 1
    return gen_sample


if __name__ == "__main__":
    ref_smiles_df = pd.read_csv('data/binders.csv').smiles.apply(standardize_smiles_error_handle).values.tolist()
    ref_fps = [_parallel_get_fps(smi) for smi in ref_smiles_df]
    get_file_names()
