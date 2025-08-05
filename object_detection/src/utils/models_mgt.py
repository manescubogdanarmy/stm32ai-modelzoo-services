#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import tensorflow as tf
from onnx import ModelProto
import onnxruntime
from omegaconf import DictConfig
import numpy as np
from tensorflow.keras import layers

from common.utils import check_model_support, check_attributes
from src.models import st_ssd_mobilenet_v1, ssd_mobilenet_v2_fpnlite, tiny_yolo_v2, st_yolo_lc_v1, \
                   st_yolo_x


def change_yolo_model_number_of_classes(model,num_classes,num_anchors):

    output_shape = (5 + num_classes)*num_anchors

    # If the model already has the correct number of classes -> dont do anything
    for outp in model.outputs:
        if outp.shape[0] == output_shape:
            return model

    l = -1
    l_list = []

    while True:

        layer_type = type(model.layers[l])
        layer_config = model.layers[l].get_config()

        if layer_type in [layers.Conv2D,
                          layers.Conv2DTranspose,
                          layers.Conv1D,
                          layers.Conv1DTranspose,
                          layers.Dense]:
            if layer_type in [layers.Conv2D,layers.Conv2DTranspose,layers.Conv1D,layers.Conv1DTranspose]:
                layer_config['filters'] = output_shape
                new_layer = layer_type(**layer_config)
                outputs = new_layer(model.layers[l-1].output)
            else:
                layer_config['units'] = output_shape
                new_layer = layer_type(**layer_config)
                outputs = new_layer(model.layers[l-1].output)

            for i,new_l in enumerate(l_list[::-1]):
                outputs = new_l(outputs)

            return tf.keras.Model(inputs=model.input, outputs=outputs, name=model.name)

        else:
            l_list.append(layer_type(**layer_config))
            l-=1

    return None


def search_layer(tensor,model,with_his="output"):
    for i,l in enumerate(model.layers): # search which layer has this tensor as output
        if not type(l)==layers.InputLayer:
            if with_his=="output":
                for lo in (l.output if type(l.output)==list else [l.output]):
                    if tensor is lo:
                        return i
            if with_his=="input":
                if tensor in (l.input if type(l.input)==list else [l.input]):
                    return i
    return None

def change_yolo_x_model_number_of_classes(model,num_classes,num_anchors):

    model_outputs = model.outputs

    concatenate_layers_indexes = []
    for o in model_outputs:
        concatenate_layers_indexes.append(search_layer(o,model,'output'))

    output_tensors_list = [] # list of output tensors of the new model

    for c in concatenate_layers_indexes: # for all Yolo_X heads

        # concatenate layer infos
        concatenate_layer_inputs = model.layers[c].input
        concatenate_layer_type   = type(model.layers[c])
        concatenate_layer_config = model.layers[c].get_config()

        new_concatenate_layer_inputs = []

        list_of_shapes = [4*num_anchors,1*num_anchors,num_classes*num_anchors]

        for i,t in enumerate(concatenate_layer_inputs):
            if t.shape[-1] != list_of_shapes[i]:

                conv_layer_index = search_layer(t,model,"output")

                conv_layer_input  = model.layers[conv_layer_index].input
                conv_layer_type   = type(model.layers[conv_layer_index])
                conv_layer_config = model.layers[conv_layer_index].get_config()

                # change the number of filters of the Conv2d layer
                conv_layer_config['filters'] = list_of_shapes[i]

                new_conv_layer = conv_layer_type(**conv_layer_config)
                new_concatenate_layer_inputs.append(new_conv_layer(conv_layer_input))
            else:
                new_concatenate_layer_inputs.append(concatenate_layer_inputs[i])

        new_concatenate_layer = concatenate_layer_type(**concatenate_layer_config)
        output_tensors_list.append(new_concatenate_layer(new_concatenate_layer_inputs))

    return tf.keras.Model(inputs=model.input, outputs=output_tensors_list, name=model.name)


def ai_runner_invoke(image_processed,ai_runner_interpreter):
    def reduce_shape(x):  # reduce shape (request by legacy API)
        old_shape = x.shape
        n_shape = [old_shape[0]]
        for v in x.shape[1:len(x.shape) - 1]:
            if v != 1:
                n_shape.append(v)
        n_shape.append(old_shape[-1])
        return x.reshape(n_shape)

    preds, _ = ai_runner_interpreter.invoke(image_processed)
    predictions = []
    for x in preds:
        x = reduce_shape(x)
        predictions.append(x.copy())
    return predictions

def model_family(model_type: str) -> str:
    if model_type in ("st_ssd_mobilenet_v1", "ssd_mobilenet_v2_fpnlite"):
        return "ssd"
    elif model_type in ("tiny_yolo_v2", "st_yolo_lc_v1"):
        return "yolo"
    elif model_type in ("yolo_v8", "yolo_v11", "yolo_v5u"):
        return "yolo_v8"
    elif model_type in ("st_yolo_x"):
        return "st_yolo_x"
    elif model_type in ("yolo_v4_tiny", "yolo_v4"):
        return "yolo_v4"
    elif model_type in ("face_detect_front"):
        return "face_detect_front"
    else:
        raise ValueError(f"Internal error: unknown model type {model_type}")


def _check_ssd_mobilenet(cft, model_type, alpha_values=None, random_resizing=None):

    check_attributes(cft, expected=["alpha", "input_shape"], optional=["pretrained_weights"], section="training.model")
                          
    message = "\nPlease check the 'training.model' section of your configuration file."
    if cft.alpha not in alpha_values:
        raise ValueError(f"\nSupported `alpha` values for `{model_type}` model are "
                         f"{alpha_values}. Received {cft.alpha}{message}")
                         
    if random_resizing:
        raise ValueError(f"\nrandom_periodic_resizing is not supported for model `{model_type}`.\n"
                         "Please check the 'data_augmentation' section of your configuration file.")

def _check_st_yolo_x(cft, model_type, random_resizing=None):

    check_attributes(cft, expected=["input_shape"], optional=["depth_mul", "width_mul"], section="training.model")
                          
    message = "\nPlease check the 'training.model' section of your configuration file."                        
    if random_resizing:
        raise ValueError(f"\nrandom_periodic_resizing is not supported for model `{model_type}`.\n"
                         "Please check the 'data_augmentation' section of your configuration file.")


def _get_zoo_model(cfg: DictConfig):
    """
    Returns a Keras model object based on the specified configuration and parameters.

    Args:
        cfg (DictConfig): A dictionary containing the configuration for the model.
        num_classes (int): The number of classes for the model.
        dropout (float): The dropout rate for the model.
        section (str): The section of the model to be used.

    Returns:
        tf.keras.Model: A Keras model object based on the specified configuration and parameters.
    """

    # Define the supported models and their versions
    supported_models = {
        'st_ssd_mobilenet_v1': None,
        'ssd_mobilenet_v2_fpnlite': None,
        'tiny_yolo_v2': None,
        'st_yolo_lc_v1': None,
        'st_yolo_x': None,

    }

    model_name = cfg.general.model_type   
    message = "\nPlease check the 'general' section of your configuration file."
    check_model_support(model_name, supported_models=supported_models, message=message)

    cft = cfg.training.model
    input_shape = cft.input_shape    
    num_classes = len(cfg.dataset.class_names)
    random_resizing = True if cfg.data_augmentation and cfg.data_augmentation.config.random_periodic_resizing else False
    section = "training.model"
    model = None

    if model_name == "st_ssd_mobilenet_v1":
        _check_ssd_mobilenet(cft, "st_ssd_mobilenet_v1",
                            alpha_values=[0.25, 0.50, 0.75, 1.0],
                            random_resizing=random_resizing)
        model = st_ssd_mobilenet_v1(input_shape, num_classes, cft.alpha, pretrained_weights=cft.pretrained_weights)
        
    elif model_name == "ssd_mobilenet_v2_fpnlite":
        _check_ssd_mobilenet(cft, "ssd_mobilenet_v2_fpnlite",
                            alpha_values=[0.35, 0.50, 0.75, 1.0],
                            random_resizing=random_resizing)
        model = ssd_mobilenet_v2_fpnlite(input_shape, num_classes, cft.alpha, pretrained_weights=cft.pretrained_weights)

    elif model_name == "tiny_yolo_v2":     
        check_attributes(cft, expected=["input_shape"], section=section)
        num_anchors = len(cfg.postprocessing.yolo_anchors)
        model = tiny_yolo_v2(input_shape, num_anchors, num_classes)

    elif model_name == "st_yolo_lc_v1":
        check_attributes(cft, expected=["input_shape"], section=section)
        num_anchors = len(cfg.postprocessing.yolo_anchors)
        model = st_yolo_lc_v1(input_shape, num_anchors, num_classes)

    elif model_name == "st_yolo_x":
        _check_st_yolo_x(cft, "st_yolo_x",random_resizing=random_resizing)
        num_anchors = len(cfg.postprocessing.yolo_anchors)
        if not cft.depth_mul and not cft.width_mul:
            cft.depth_mul = 0.33
            cft.width_mul = 0.25
        model = st_yolo_x(input_shape=input_shape, num_anchors=num_anchors, num_classes=num_classes, depth_mul=cft.depth_mul, width_mul=cft.width_mul)

    return model
    

def load_model_for_training(cfg: DictConfig) -> tuple:
    """"
    Loads a model for training.
    
    The model to train can be:
    - a model from the Model Zoo
    - a user model (BYOM)
    - a model previously trained during a training that was interrupted.
    
    When a training is run, the following files are saved in the saved_models
    directory:
        base_model.h5:
            Model saved before the training started. Weights are random.
        best_weights.h5:
            Best weights obtained since the beginning of the training.
        last_weights.h5:
            Weights saved at the end of the last epoch.
    
    To resume a training, the last weights are loaded into the base model.
    """
    
    model_type = cfg.general.model_type    
    model = None
    
    # Train a model from the Model Zoo
    if cfg.training.model:
        print("[INFO] : Loading Model Zoo model:", model_type)        
        model = _get_zoo_model(cfg)
        
        cft = cfg.training.model
        if cft.pretrained_weights:
            print(f"[INFO] : Loaded pretrained weights: `{cft.pretrained_weights}`")
        else:
            print(f"[INFO] : No pretrained weights were loaded.")
        
    # Bring your own model
    elif cfg.general.model_path:
        print("[INFO] : Loading model", cfg.general.model_path)
        model = tf.keras.models.load_model(cfg.general.model_path, compile=False)
        
        if cfg.general.model_type in ["tiny_yolo_v2","st_yolo_lc_v1"]:
            model = change_yolo_model_number_of_classes(model,num_classes=len(cfg.dataset.class_names),num_anchors=len(cfg.postprocessing.yolo_anchors))

        if cfg.general.model_type in ["st_yolo_x"]:
            model = change_yolo_x_model_number_of_classes(model,num_classes=len(cfg.dataset.class_names),num_anchors=len(cfg.postprocessing.yolo_anchors))
            
        # Check that the model has a specified input shape
        input_shape = tuple(model.input.shape[1:])
        if None in input_shape:
            raise ValueError(f"\nThe model input shape is unspecified. Got {str(input_shape)}\n"
                             "Unable to proceed with training.")
                        
    # Resume a previously interrupted training
    elif cfg.training.resume_training_from:
        resume_dir = os.path.join(cfg.training.resume_training_from, cfg.general.saved_models_dir)
        print(f"[INFO] : Resuming training from directory {resume_dir}\n")
        
        message = "\nUnable to resume training."
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"\nCould not find resume directory {resume_dir}{message}")
        model_path = os.path.join(resume_dir, "base_model.h5")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"\nCould not find model file {model_path}{message}\n")
        last_weights_path = os.path.join(resume_dir, "last_weights.h5")
        if not os.path.isfile(last_weights_path):
            raise FileNotFoundError(f"\nCould not find model weights file {last_weights_path}{message}\n")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.load_weights(last_weights_path)

    return model
