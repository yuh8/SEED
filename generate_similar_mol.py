import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticHeterocycles
from copy import deepcopy
from multiprocessing import Pool, freeze_support
from src.base_model_utils import SeedGenerator
from src.data_process_utils import (get_last_col_with_atom,
                                    get_initial_act_vec,
                                    draw_smiles,
                                    graph_to_smiles, smiles_to_graph,
                                    standardize_smiles_error_handle,
                                    decompose_smi_graph)
from src.reward_utils import (get_logp_reward, get_sa_reward,
                              get_qed_reward, get_cycle_reward)
from src.misc_utils import create_folder, load_json_model, sample_action
from src.CONSTS import (BOND_NAMES, MAX_NUM_ATOMS, MAX_GEN_ATOMS,
                        MIN_NUM_ATOMS, FEATURE_DEPTH,
                        ATOM_MAX_VALENCE,
                        ATOM_LIST, CHARGES)
np.set_printoptions(linewidth=10000)
np.set_printoptions(threshold=sys.maxsize)


def get_initial_smi_graph(smiles):
    smi_graph = smiles_to_graph(smiles)
    return smi_graph


def check_validity(mol):
    try:
        Chem.SanitizeMol(mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def update_state_with_action(action_logits, state, num_atoms):
    is_terminate = False
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    max_remaining_valence = state[:, :, -1].sum(-1).max()
    col = get_last_col_with_atom(state)
    col_has_atom = (state[:, col, :-1].sum(-1) > 0).any()
    if (max_remaining_valence < 1) and (col > 0):
        is_terminate = True
        return state, is_terminate

    feature_vec = np.zeros(FEATURE_DEPTH)
    action_idx = sample_action(action_logits, state)

    if action_idx <= num_act_charge_actions:
        if col >= num_atoms:
            is_terminate = True
            return state, is_terminate

        atom_idx = action_idx // len(CHARGES)
        charge_idx = action_idx % len(CHARGES) - len(CHARGES)
        feature_vec[atom_idx] = 2
        feature_vec[charge_idx] = 1

        if col_has_atom:
            state[col + 1, col + 1, :-1] = feature_vec
            # once an atom is added, initialize with full valence
            state[col + 1, col + 1, -1] = ATOM_MAX_VALENCE[atom_idx]
        else:
            state[col, col, :-1] = feature_vec
            state[col, col, -1] = ATOM_MAX_VALENCE[atom_idx]
    else:
        row = (action_idx - num_act_charge_actions) // len(BOND_NAMES)
        bond_idx = (action_idx - num_act_charge_actions) % len(BOND_NAMES)
        bond_feature_idx = len(ATOM_LIST) + bond_idx
        atom_idx_row = state[row, row, :len(ATOM_LIST)].argmax()
        atom_idx_col = state[col, col, :len(ATOM_LIST)].argmax()
        feature_vec[bond_feature_idx] = 1
        feature_vec[atom_idx_row] += 1
        feature_vec[atom_idx_col] += 1
        state[row, col, :-1] = feature_vec
        state[col, row, :-1] = feature_vec
        state[row, row, -1] -= bond_idx
        state[col, col, -1] -= bond_idx

    return state, is_terminate


def update_state_with_action_validity_check(action_logits, state,
                                            initial_col, num_atoms,
                                            forbid_atom_idx=[], must_add_bond_idx=[]):
    is_terminate = False
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    max_remaining_valence = state[:, :, -1].sum(-1).max()
    col = get_last_col_with_atom(state)
    col_has_atom = (state[:, col, :-1].sum(-1) > 0).any()
    if (max_remaining_valence < 1) and (col > 0):
        print('no valence available')
        is_terminate = True
        return state, is_terminate

    valid = False
    while not valid:
        feature_vec = np.zeros(FEATURE_DEPTH)
        try:
            action_idx = sample_action(action_logits, state, initial_col, forbid_atom_idx, must_add_bond_idx)
        except:
            is_terminate = True
            state[col, col] = 0
            return state, is_terminate
        state_new = deepcopy(state)
        if action_idx <= num_act_charge_actions:
            if col >= num_atoms:
                is_terminate = True
                return state, is_terminate

            atom_idx = action_idx // len(CHARGES)
            charge_idx = action_idx % len(CHARGES) - len(CHARGES)
            feature_vec[atom_idx] = 2
            feature_vec[charge_idx] = 1

            if col_has_atom:
                state_new[col + 1, col + 1, :-1] = feature_vec
                # once an atom is added, initialize with full valence
                state_new[col + 1, col + 1, -1] = ATOM_MAX_VALENCE[atom_idx]
            else:
                state_new[col, col, :-1] = feature_vec
                state_new[col, col, -1] = ATOM_MAX_VALENCE[atom_idx]
            return state_new, is_terminate
        else:
            row = (action_idx - num_act_charge_actions) // len(BOND_NAMES)
            bond_idx = (action_idx - num_act_charge_actions) % len(BOND_NAMES)
            bond_feature_idx = len(ATOM_LIST) + bond_idx
            atom_idx_row = state_new[row, row, :len(ATOM_LIST)].argmax()
            atom_idx_col = state_new[col, col, :len(ATOM_LIST)].argmax()
            feature_vec[bond_feature_idx] = 1
            feature_vec[atom_idx_row] += 1
            feature_vec[atom_idx_col] += 1
            state_new[row, col, :-1] = feature_vec
            state_new[col, row, :-1] = feature_vec
            state_new[row, row, -1] -= bond_idx
            state_new[col, col, -1] -= bond_idx
            mol = graph_to_smiles(state_new[:, :, :-1], return_mol=True)
            valid = check_validity(mol)
            if not valid:
                action_logits[action_idx] = -1e9

    return state_new, is_terminate


def check_symmetric(a, rtol=1e-10, atol=1e-10):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def generate_smiles(model, initial_state, min_num_atoms):
    num_atoms = np.random.randint(min_num_atoms, 13)
    # state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    state = deepcopy(initial_state)
    initial_col = get_last_col_with_atom(initial_state)
    is_terminate = False
    # forbid_atom_idx = [0, 1, 2, 4, 6]
    # must_add_bond_idx = [3, 5]
    # must_add_bond_idx = []
    # forbid_atom_indices = [[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
    #                        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
    #                        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
    #                        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
    must_add_bond_idx = []
    forbid_atom_idx = [0, 2, 3, 4, 5]

    while not is_terminate:
        X_in = deepcopy(state[np.newaxis, ...])
        X_in[..., -1] /= 8
        action_logits = model(X_in, training=False).numpy()[0]
        state, is_terminate = update_state_with_action_validity_check(action_logits, state,
                                                                      initial_col, num_atoms,
                                                                      forbid_atom_idx,
                                                                      must_add_bond_idx)
        # state, is_terminate = update_state_with_action(action_logits, state, num_atoms)

    smi_graph = state[..., :-1]
    smi = graph_to_smiles(smi_graph)
    smi = _canonicalize_smiles(smi)
    if not isinstance(smi, str):
        return None
    # draw_smiles(smi, "gen_samples_rl/gen_sample_{}".format(gen_idx))
    return smi, num_atoms


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


def get_initial_state_from_smiles(smi):
    smi_graph = smiles_to_graph(smi)
    _, states = decompose_smi_graph(smi_graph)
    initial_state = states[-1]
    return initial_state


if __name__ == "__main__":
    freeze_support()
    create_folder('gen_samples_rl/')
    create_folder('gen_samples_rl_similar_binder/')
    model = load_json_model("base_model/generator_model.json", SeedGenerator, "SeedGenerator")
    model.compile(optimizer='Adam')
    model.load_weights("./base_model/weights/")
    gen_samples_df = []
    gen_samples_good_df = []
    count = 0
    count_good = 0
    mode = 'diversity'
    initial_smi = 'C1=NC=CC=C1'
    # initial_smi = 'C1=CC(O)=CC=C1'
    smi = standardize_smiles_error_handle(initial_smi)
    mol = Chem.MolFromSmiles(smi)
    min_num_atoms = mol.GetNumAtoms() + 2
    ref_smiles_df = pd.read_csv('data/binders.csv').smiles.apply(standardize_smiles_error_handle).values.tolist()
    ref_fps = [_parallel_get_fps(smi) for smi in ref_smiles_df]
    initial_state = get_initial_state_from_smiles(initial_smi)
    smis = []
    for idx in range(1000000):
        gen_sample = {}
        gen_sample_good = {}
        try:
            smi, num_atoms = generate_smiles(model, initial_state, min_num_atoms)
        except Exception as e:
            print(e)
            continue

        if 's' in smi or 'S' in smi or '[nH]' in smi or '+' in smi or '[O-]' in smi:
            continue

        gen_sample["Smiles"] = smi
        gen_sample["NumAtoms"] = num_atoms
        gen_sample['logp'] = np.round(get_logp_reward(smi), 4)
        gen_sample['sa'] = np.round(get_sa_reward(smi), 4)
        gen_sample['cycle'] = get_cycle_reward(smi)
        gen_sample['qed'] = np.round(get_qed_reward(smi), 4)
        gen_samples_df.append(gen_sample)
        count += 1
        fp = _parallel_get_fps(smi)
        sim = _parallel_get_ext_dists(fp, ref_fps)
        print("validation rate = {0} and QED = {1} and SA = {2} and sim = {3}".format(np.round(count / (idx + 1), 3),
                                                                                      get_qed_reward(smi),
                                                                                      get_sa_reward(smi), sim))
        # draw_smiles(smi, "gen_samples_rl/gen_sample_good_{}".format(idx))
        mol = Chem.MolFromSmiles(smi)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        mol_w = MolWt(mol)
        sim, sim_idx = _parallel_get_ext_dists(fp, ref_fps)
        num_ofht_cycles = CalcNumAromaticHeterocycles(mol)
        if sim > 0.2 and mol_w < 300 and np.round(get_sa_reward(smi), 4) < 4 and num_ofht_cycles > 1:
            # if sim > 0.2 and mol_w < 250 and np.round(get_sa_reward(smi), 4) < 4:
            if smi in smis:
                continue
            smis.append(smi)
            gen_sample_good["Smiles"] = smi
            gen_sample_good["NumAtoms"] = num_atoms
            gen_sample_good['logp'] = np.round(get_logp_reward(smi), 4)
            gen_sample_good['sa'] = np.round(get_sa_reward(smi), 4)
            gen_sample_good['cycle'] = get_cycle_reward(smi)
            gen_sample_good['qed'] = np.round(get_qed_reward(smi), 4)
            gen_sample_good['similarity'] = sim
            gen_sample_good['mol_weight'] = mol_w
            gen_sample_good['matched'] = sim_idx + 1
            gen_samples_good_df.append(gen_sample_good)
            draw_smiles(smi, f"gen_samples_rl_similar_binder/{smi}")
            count_good += 1
            if len(gen_samples_good_df) >= 1000:
                break

    gen_samples_df = pd.DataFrame(gen_samples_df)
    gen_samples_df.sort_values(by=['qed'], inplace=True, ascending=False)
    gen_samples_df.to_csv('generated_molecules_rl.csv', index=False)

    gen_samples_good_df = pd.DataFrame(gen_samples_good_df)
    gen_samples_good_df.to_csv('generated_molecules_similar_binder.csv', index=False)
    breakpoint()
