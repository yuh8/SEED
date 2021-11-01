import os
import json
import tensorflow as tf


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
