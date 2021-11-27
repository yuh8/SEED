import numpy as np
from rdkit import Chem
from copy import deepcopy
from .data_process_utils import graph_to_smiles
from .reward_utils import (get_penalized_logp_reward,
                           get_sa_reward,
                           get_qed_reward)
from .CONSTS import (MIN_NUM_ATOMS, BOND_NAMES,
                     QED_WEIGHT, SA_WEIGHT,
                     MAX_NUM_ATOMS,
                     FEATURE_DEPTH,
                     ATOM_MAX_VALENCE,
                     ATOM_LIST, CHARGES)


def check_validity(mol):
    try:
        Chem.SanitizeMol(mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def get_last_col_with_atom(state):
    # last column with atom
    col_state = state.sum(-1).sum(-1)
    col = np.maximum(col_state[col_state > 0].shape[0] - 1, 0)
    return col


def update_state_with_action(action_idx, state, num_atoms):
    '''
    return state, statue, intermediate reward
    '''
    is_terminate = False
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    max_remaining_valence = state[:, :, -1].sum(-1).max()
    col = get_last_col_with_atom(state)
    col_has_atom = (state[:, col, :-1].sum(-1) > 0).any()
    if (max_remaining_valence < 2) and (col > 0):
        is_terminate = True
        return state, is_terminate, 0

    feature_vec = np.zeros(FEATURE_DEPTH)
    state_new = deepcopy(state)
    if action_idx <= num_act_charge_actions:
        if col >= num_atoms:
            is_terminate = True
            return state, is_terminate, 0

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
        reward = 0
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
            is_terminate = True
            # if bond creation is not valid, add small penalty
            return state_new, is_terminate, -1
        reward = 0

    return state_new, is_terminate, reward


class Env:
    def __init__(self, num_atoms, mode='QED'):
        self.reset()
        self.num_atoms = num_atoms
        self.mode = mode

    def step(self, action_idx):
        self.state, done, inter_reward = update_state_with_action(action_idx, self.state, self.num_atoms)
        if not done:
            return self.state, done, inter_reward

        smi = graph_to_smiles(self.state[:, :, :-1])

        if self.mode == "QED":
            final_r = QED_WEIGHT * get_qed_reward(smi)  # + SA_WEIGHT * get_sa_reward(smi) / 10
        else:
            final_r = np.exp(get_penalized_logp_reward(smi) / 4)

        col = get_last_col_with_atom(self.state)

        if col <= MIN_NUM_ATOMS:
            final_r = -1

        # normalize reward between -1 and 1
        final_r += inter_reward  # + get_diversity_reward(smi)
        # final_r /= (QED_WEIGHT + 1)
        return self.state, done, final_r

    def reset(self):
        self.state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
        self.is_terminate = False
        return self.state
