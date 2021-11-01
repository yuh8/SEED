import numpy as np
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from .CONSTS import (FEATURE_DEPTH, MAX_NUM_ATOMS,
                     ATOM_LIST, ATOM_MAX_VALENCE,
                     BOND_NAMES, CHARGES)
RDLogger.DisableLog('rdApp.*')


def has_valid_elements(mol):
    len_elements = len([atom.GetSymbol() for atom in mol.GetAtoms()])
    if len_elements > MAX_NUM_ATOMS:
        return False

    has_unknown_element = [atom.GetSymbol() not in ATOM_LIST for atom in mol.GetAtoms()]
    if sum(has_unknown_element) > 0:
        return False

    has_unknown_charge = [atom.GetFormalCharge() not in CHARGES for atom in mol.GetAtoms()]
    if sum(has_unknown_charge) > 0:
        return False

    has_radical = [atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()]
    if sum(has_radical) > 0:
        return False

    return True


def is_smile_valid(smi):
    try:
        if Chem.MolFromSmiles(smi) is None:
            return False
    except:
        return False

    mol = Chem.MolFromSmiles(smi)
    if not has_valid_elements(mol):
        return False

    return True


def is_mol_valid(mol):
    try:
        Chem.MolToSmiles(mol)
    except:
        return False

    if not has_valid_elements(mol):
        return False

    return True


def standardize_smiles(smi):
    '''
    convert smiles to Kekulized form
    to convert aromatic bond to single/double/triple bond
    '''
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    return smi


def standardize_smiles_to_mol(smi):
    '''
    remove aromatic bonds in mol object
    '''
    smi = standardize_smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except:
        return mol
    return mol


def draw_smiles(smi, file_name):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Draw.MolToFile(mol, '{}.png'.format(file_name))


def smiles_to_graph(smi):
    if not is_smile_valid(smi):
        return None
    mol = standardize_smiles_to_mol(smi)
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    for ii, ei in enumerate(elements):
        for jj, ej in enumerate(elements):
            feature_vec = np.zeros(FEATURE_DEPTH)
            if ii == jj:
                charge_idx = CHARGES.index(charges[ii]) - len(CHARGES)
                feature_vec[charge_idx] = 1

            if ii > jj:
                continue

            atom_idx_ii = ATOM_LIST.index(ei)
            atom_idx_jj = ATOM_LIST.index(ej)
            feature_vec[atom_idx_ii] += 1
            feature_vec[atom_idx_jj] += 1
            if mol.GetBondBetweenAtoms(ii, jj) is not None:
                bond_name = mol.GetBondBetweenAtoms(ii, jj).GetBondType()
                bond_idx = BOND_NAMES.index(bond_name)
                bond_feature_idx = len(ATOM_LIST) + bond_idx
                feature_vec[bond_feature_idx] = 1
            smi_graph[ii, jj, :] = feature_vec
            smi_graph[jj, ii, :] = feature_vec

    return smi_graph


def update_atom_property(mol, charges):
    for key in charges:
        mol.GetAtomWithIdx(key).SetFormalCharge(int(charges[key]))
    return mol


def graph_to_smiles(smi_graph, return_mol=False):
    con_graph = np.sum(smi_graph, axis=-1)
    graph_dim = con_graph.shape[0]
    mol = Chem.RWMol()
    atoms = {}
    charges = {}
    for ii in range(graph_dim):
        if con_graph[ii, ii] == 3:
            atom_feature_vec = smi_graph[ii, ii, :len(ATOM_LIST)]
            charge_feature_vec = smi_graph[ii, ii, -len(CHARGES):]
            atom = np.array(ATOM_LIST)[atom_feature_vec.argmax()]
            atom = Chem.Atom(atom)
            atom_idx = mol.AddAtom(atom)
            atoms[ii] = atom_idx
            charges[atom_idx] = np.array(CHARGES)[charge_feature_vec.argmax()]

    for ii in range(graph_dim):
        for jj in range(graph_dim):
            if ii >= jj:
                continue

            if (con_graph[ii, jj] == 3) and \
                    (ii in atoms.keys()) and (jj in atoms.keys()):
                bond_feature_vec = smi_graph[ii, jj, len(ATOM_LIST):-len(CHARGES)]
                bond_type = BOND_NAMES[bond_feature_vec.argmax()]
                mol.AddBond(atoms[ii], atoms[jj], bond_type)

    mol = update_atom_property(mol, charges)
    if return_mol:
        return mol
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    return smi


def get_initial_act_vec():
    num_atom_actions = len(ATOM_LIST)
    num_charge_actions = len(CHARGES)
    num_act_charge_actions = num_atom_actions * num_charge_actions
    # number of location to place atoms x num of bond types
    num_loc_bond_actions = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
    action_vec = np.zeros(num_act_charge_actions + num_loc_bond_actions)
    return action_vec


def act_idx_to_vect(action_idx):
    action_vec = get_initial_act_vec()
    atom_act_idx, charge_act_idx = action_idx[0]
    loc_act_idx, bond_act_idx = action_idx[1]
    if atom_act_idx is not None and charge_act_idx is not None:
        dest_idx = atom_act_idx * len(CHARGES) + charge_act_idx
        action_vec[dest_idx] = 1

    if loc_act_idx is not None and bond_act_idx is not None:
        start_idx = len(ATOM_LIST) * len(CHARGES)
        dest_idx = start_idx + loc_act_idx * len(BOND_NAMES) + bond_act_idx
        action_vec[dest_idx] = 1
    return action_vec


def get_last_col_with_atom(state):
    # last column with atom
    col_state = state.sum(-1).sum(-1)
    col = np.maximum(col_state[col_state > 0].shape[0] - 1, 0)
    return col


def get_action_mask_from_state(state):
    '''
    indicate masking location with value 1
    '''
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    action_vec_mask = get_initial_act_vec()

    # masking all zero bond creation action
    zero_bond_idx = np.arange(num_act_charge_actions,
                              action_vec_mask.shape[0],
                              len(BOND_NAMES))
    action_vec_mask[zero_bond_idx] = 1

    # bond creation allowed only in upper triangle
    col = get_last_col_with_atom(state)
    action_vec_mask[num_act_charge_actions + col * len(BOND_NAMES):] = 1

    # zero or single atom state allows only atom creation
    if col == 0:
        return action_vec_mask

    # if no bond has been created for this col,
    # do not allow atom_charge creation
    col_has_bond = state[:col, col].sum() > 0
    if not col_has_bond:
        action_vec_mask[:num_act_charge_actions] = 1

    # scan until the row above the col number
    start_idx = num_act_charge_actions
    remaining_valence_col = state[col, col, -1]
    for row in range(col):
        remaining_valence_row = state[row, row, -1]
        min_remaining_valence = np.minimum(remaining_valence_row,
                                           remaining_valence_col)
        cell_has_bond = state[row, col, :-1].sum(-1) == 3
        # if cell has bond, do not allow bond creation
        if cell_has_bond:
            mask_start = start_idx + row * len(BOND_NAMES)
            mask_end = start_idx + (row + 1) * len(BOND_NAMES)
            action_vec_mask[mask_start:mask_end] = 1
            continue

        # do not allow creation of bond exceeding minimum remaining valance
        has_invalid_bond = (np.arange(len(BOND_NAMES)) > min_remaining_valence).any()
        if has_invalid_bond:
            mask_start = int(start_idx + row * len(BOND_NAMES) + min_remaining_valence + 1)
            mask_end = int(start_idx + (row + 1) * len(BOND_NAMES))
            action_vec_mask[mask_start:mask_end] = 1

    return action_vec_mask


def decompose_smi_graph(smi_graph):
    gragh_dim = smi_graph.shape[0]
    # last feature dim tracks the remaining valence
    state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    states = [deepcopy(state)]
    actions = []
    for jj in range(gragh_dim):
        # terminate
        if sum(smi_graph[jj, jj, :]) == 0:
            return actions, states[:-1]
        # adding atom and charge
        atom_act_idx = smi_graph[jj, jj, :len(ATOM_LIST)].argmax()
        charge_act_idx = smi_graph[jj, jj, -len(CHARGES):].argmax()
        loc_act_idx = None
        bond_act_idx = None
        action_idx = ((atom_act_idx, charge_act_idx), (loc_act_idx, bond_act_idx))
        actions.append(act_idx_to_vect(action_idx))
        state[jj, jj, :-1] = smi_graph[jj, jj, :]
        # once an atom is added, initialize with full valence
        state[jj, jj, -1] = ATOM_MAX_VALENCE[atom_act_idx]
        states.append(deepcopy(state))
        for ii in range(jj):
            charge_act_idx = None
            atom_act_idx = None
            if sum(smi_graph[ii, jj, :]) == 3:
                # adding connection bond
                loc_act_idx = ii
                bond_act_idx = smi_graph[ii, jj, len(ATOM_LIST):-len(CHARGES)].argmax()
                action_idx = ((atom_act_idx, charge_act_idx), (loc_act_idx, bond_act_idx))
                actions.append(act_idx_to_vect(action_idx))
                # ensure symmetry
                state[ii, jj, :-1] = smi_graph[ii, jj, :]
                state[jj, ii, :-1] = smi_graph[ii, jj, :]
                # once a bond is added, deduct valence with bond_idx for connected atoms
                state[ii, ii, -1] -= bond_act_idx
                state[jj, jj, -1] -= bond_act_idx
                states.append(deepcopy(state))

    return actions, states[:-1]


if __name__ == "__main__":
    smi = 'CCC(C(O)C)CN'
    smiles_to_graph(smi)
