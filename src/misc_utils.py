import os
import json
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from .CONSTS import (ATOM_LIST,
                     CHARGES, BOND_NAMES,
                     MAX_NUM_ATOMS)
from .data_process_utils import (get_action_mask_from_state,
                                 get_initial_act_vec)


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def save_model_to_json(model, model_path):
    model_json = model.to_json()
    with open("{}".format(model_path), "w") as json_file:
        json.dump(model_json, json_file)


def load_json_model(model_path, custom_obj=None, custom_obj_name=None):
    with open("{}".format(model_path)) as json_file:
        model_json = json.load(json_file)
    if custom_obj is not None:
        uncompiled_model = tf.keras.models.model_from_json(model_json,
                                                           {custom_obj_name: custom_obj})
    else:
        uncompiled_model = tf.keras.models.model_from_json(model_json)
    return uncompiled_model


def logprobabilities(logits, a):
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    num_loc_bond_actions = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
    num_actions = num_act_charge_actions + num_loc_bond_actions
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


def sample_action(action_logits, state, T=1):
    action_mask = get_action_mask_from_state(state)
    action_logits = np.where(action_mask, -1e9, action_logits)
    action_probs = softmax(action_logits / T)
    act_vec = get_initial_act_vec()
    action_size = act_vec.shape[0]
    action_idx = np.random.choice(action_size, p=action_probs)
    return action_idx
