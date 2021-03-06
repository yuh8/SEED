import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from copy import deepcopy
from multiprocessing import Pool, freeze_support
from src.base_model_utils import SeedGenerator
from src.data_process_utils import (get_last_col_with_atom,
                                    draw_smiles,
                                    graph_to_smiles,
                                    standardize_smiles_error_handle)
from src.reward_utils import (get_logp_reward, get_sa_reward,
                              get_qed_reward, get_cycle_reward)
from src.misc_utils import create_folder, load_json_model, sample_action
from src.CONSTS import (BOND_NAMES, MAX_NUM_ATOMS, MAX_GEN_ATOMS,
                        MIN_NUM_ATOMS, FEATURE_DEPTH,
                        ATOM_MAX_VALENCE,
                        ATOM_LIST, CHARGES)
np.set_printoptions(linewidth=10000)
np.set_printoptions(threshold=sys.maxsize)


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
    if (max_remaining_valence < 2) and (col > 0):
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


def update_state_with_action_validity_check(action_logits, state, num_atoms):
    is_terminate = False
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    max_remaining_valence = state[:, :, -1].sum(-1).max()
    col = get_last_col_with_atom(state)
    col_has_atom = (state[:, col, :-1].sum(-1) > 0).any()
    if (max_remaining_valence < 2) and (col > 0):
        is_terminate = True
        return state, is_terminate

    valid = False
    while not valid:
        feature_vec = np.zeros(FEATURE_DEPTH)
        try:
            action_idx = sample_action(action_logits, state)
        except:
            is_terminate = True
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


def generate_smiles(model, gen_idx):
    num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_GEN_ATOMS)
    state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    is_terminate = False

    while not is_terminate:
        X_in = state[np.newaxis, ...]
        action_logits = model(X_in, training=False).numpy()[0]
        state, is_terminate = update_state_with_action_validity_check(action_logits, state, num_atoms)
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


def compute_unique_score():
    gen_samples_df = pd.read_csv("generated_molecules_rl.csv")
    gen_samples_df.loc[:, 'CanSmiles'] = gen_samples_df.Smiles.map(_canonicalize_smiles)
    gen_samples_df = gen_samples_df[~gen_samples_df.CanSmiles.isnull()]
    num_uniques = gen_samples_df.CanSmiles.unique().shape[0]
    unique_score = np.round(num_uniques / gen_samples_df.shape[0], 3)
    print("Unique score = {}".format(unique_score))
    return unique_score


def compute_novelty_score():
    gen_samples_df = pd.read_csv("generated_molecules_rl.csv")
    train_samples_df = pd.read_csv('D:/seed_data/generator/train_data/df_train.csv')
    gen_samples_df.loc[:, 'CanSmiles'] = gen_samples_df.Smiles.map(_canonicalize_smiles)
    train_samples_df.loc[:, 'CanSmiles'] = train_samples_df.Smiles.map(_canonicalize_smiles)
    gen_samples_df = gen_samples_df[~gen_samples_df.CanSmiles.isnull()]
    train_samples_df = train_samples_df[~train_samples_df.CanSmiles.isnull()]
    gen_smi_unique = list(gen_samples_df.CanSmiles.unique())
    train_smi_unique = list(train_samples_df.CanSmiles.unique())
    intersection_samples = list(set(gen_smi_unique) & set(train_smi_unique))
    novelty_score = np.round(1 - len(intersection_samples) / len(gen_smi_unique), 3)
    print("Novelty score = {}".format(novelty_score))
    return novelty_score


def _parallel_get_fps(smi):
    smi = _canonicalize_smiles(smi)
    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 6, 2048)


def _parallel_get_int_dists(fp_new, fps):
    dist = DataStructs.BulkTanimotoSimilarity(fp_new, fps)
    return np.sum(dist)


def _parallel_get_ext_dists(fp_new, fps):
    sm = DataStructs.BulkTanimotoSimilarity(fp_new, fps)
    return np.max(sm)


def compute_external_diversity(file_A, file_B):
    smiles_A = pd.read_csv(file_A).Smiles.map(standardize_smiles_error_handle)
    smiles_B = pd.read_csv(file_B).iloc[:1000000].Smiles
    smiles_B = smiles_B.map(standardize_smiles_error_handle)
    smiles_A = smiles_A[~smiles_A.isnull()]
    smiles_B = smiles_B[~smiles_B.isnull()]

    with Pool() as pool:
        fps_A = pool.map(_parallel_get_fps, smiles_A)

    with Pool() as pool:
        fps_B = pool.map(_parallel_get_fps, smiles_B)

    fps_A = list(filter(None, fps_A))
    fps_B = list(filter(None, fps_B))

    sims = []
    for fp in fps_A:
        simi = _parallel_get_ext_dists(fp, fps_B)
        sims.append(simi)

    return 1 - np.sum(sims) / len(fps_A)


def compute_internal_diversity(file_A):
    smiles_A = pd.read_csv(file_A).Smiles.map(standardize_smiles_error_handle)
    smiles_A = smiles_A[~smiles_A.isnull()]

    with Pool() as pool:
        fps_A = pool.map(_parallel_get_fps, smiles_A)

    fps_A = list(filter(None, fps_A))

    sims = []
    for fp in fps_A:
        simi = _parallel_get_int_dists(fp, fps_A)
        sims.append(simi)

    return 1 - np.sum(sims) / (len(fps_A) ** 2)


if __name__ == "__main__":
    freeze_support()
    create_folder('gen_samples_rl/')
    model = load_json_model("rl_model_2021-12-26/rl_model.json", SeedGenerator, "SeedGenerator")
    model.compile(optimizer='Adam')
    model.load_weights("./rl_model_2021-12-26/weights/")
    gen_samples_df = []
    gen_samples_good_df = []
    count = 0
    count_good = 0
    mode = 'diversity'
    for idx in range(10000):
        gen_sample = {}
        gen_sample_good = {}
        try:
            smi, num_atoms = generate_smiles(model, idx)
        except Exception as e:
            print(e)
            continue

        gen_sample["Smiles"] = smi
        gen_sample["NumAtoms"] = num_atoms
        gen_sample['logp'] = np.round(get_logp_reward(smi), 4)
        gen_sample['sa'] = np.round(get_sa_reward(smi), 4)
        gen_sample['cycle'] = get_cycle_reward(smi)
        gen_sample['qed'] = np.round(get_qed_reward(smi), 4)
        gen_samples_df.append(gen_sample)
        count += 1
        print("validation rate = {0} and QED = {1} and SA = {2}".format(np.round(count / (idx + 1), 3),
                                                                        get_qed_reward(smi),
                                                                        get_sa_reward(smi)))
        # draw_smiles(smi, "gen_samples_rl/gen_sample_good_{}".format(idx))
        if np.round(get_qed_reward(smi), 4) > 0.85 and np.round(get_sa_reward(smi), 4) <= 3.5:
            gen_sample_good["Smiles"] = smi
            gen_sample_good["NumAtoms"] = num_atoms
            gen_sample_good['logp'] = np.round(get_logp_reward(smi), 4)
            gen_sample_good['sa'] = np.round(get_sa_reward(smi), 4)
            gen_sample_good['cycle'] = get_cycle_reward(smi)
            gen_sample_good['qed'] = np.round(get_qed_reward(smi), 4)
            gen_samples_good_df.append(gen_sample_good)
            draw_smiles(smi, "gen_samples_rl/gen_sample_good_{}".format(count_good))
            count_good += 1
            if len(gen_samples_good_df) >= 5000:
                break

    gen_samples_df = pd.DataFrame(gen_samples_df)
    gen_samples_df.sort_values(by=['qed'], inplace=True, ascending=False)
    gen_samples_df.to_csv('generated_molecules_rl.csv', index=False)

    gen_samples_good_df = pd.DataFrame(gen_samples_good_df)
    gen_samples_good_df.to_csv('generated_molecules_good_rl.csv', index=False)
    if mode == "diversity":
        ext_div = compute_external_diversity('generated_molecules_rl.csv', 'generated_molecules_chembl.csv')
        int_div = compute_internal_diversity('generated_molecules_rl.csv')
        print('external diversity = {0} and internal diversity = {1}'.format(ext_div, int_div))
    compute_unique_score()
    compute_novelty_score()
    breakpoint()
