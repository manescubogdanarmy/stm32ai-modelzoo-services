# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import tensorflow as tf
from typing import Optional
from omegaconf import DictConfig
from src.preprocessing import preprocess_image, preprocess_input, read_image
from src.postprocessing import postprocess


def predict(cfg: Optional[DictConfig] = None) -> None:
    """
    Run inference on all the images within the test set.

    Args:
        cfg (DictConfig): The configuration file.

    Returns:
        None
    """
    model_path = cfg.general.model_path
    test_images_dir = cfg.prediction.test_files_path
    cpp = cfg.preprocessing
    channels = 1 if cpp.color_mode == "grayscale" else 3
    aspect_ratio = cpp.resizing.aspect_ratio
    interpolation = cpp.resizing.interpolation
    scale = cpp.rescaling.scale
    offset = cpp.rescaling.offset
    diameter = cfg.postprocessing.bilateral_diameter
    sigma_color = cfg.postprocessing.sigma_color
    sigma_space = cfg.postprocessing.sigma_space
    alpha = cfg.postprocessing.scaling_alpha
    beta  = cfg.postprocessing.scaling_beta
    saturation_scale = cfg.postprocessing.saturation_scale
    prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')

    if not Path(model_path).suffix == ".tflite":
        raise RuntimeError("Evaluation internal error: unsupported model type")

    print("[INFO] Making predictions using:")
    print(f"  Model: {model_path}")
    print(f"  Images directory: {test_images_dir}")

    # Load the TFLite model and allocate tensors
    net = tf.lite.Interpreter(model_path=model_path)
    net.allocate_tensors()
    input_details = net.get_input_details()[0]
    input_index_quant = input_details["index"]
    output_index_quant = net.get_output_details()
    height, width, _ = input_details['shape_signature'][1:]

    for file in os.listdir(test_images_dir):
        if file.endswith(".jpg"):
            image_path = os.path.join(test_images_dir, file)
            img = read_image(image_path, channels)
            img_process = preprocess_image(img, height, width, aspect_ratio, interpolation, scale, offset)
            img_process = preprocess_input(img_process, input_details)
            net.set_tensor(input_index_quant, img_process)
            net.invoke()

            generated_tensor = net.get_tensor(output_index_quant[0]["index"])

            postprocess(generated_tensor, diameter, sigma_color, sigma_space,
                alpha, beta, saturation_scale, prediction_result_dir, file)