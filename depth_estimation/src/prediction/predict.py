# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
import tensorflow as tf
import onnxruntime
from matplotlib import gridspec, pyplot as plt
import warnings
from pathlib import Path
from omegaconf import DictConfig
import cv2

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from common.utils import get_model_name_and_its_input_shape, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.evaluation import predict_onnx
from src.utils import ai_runner_invoke
from src.preprocessing import preprocess_image, preprocess_input, postprocess_output_values
import onnxruntime


def postprocess_output_values(output, output_details=None):
    output = np.squeeze(output)
    output = output.astype(np.float32)
    output = output - np.min(output)
    if np.max(output) > 0:
        output = output / np.max(output)
        output = cv2.GaussianBlur(output, ksize =(3,3),sigmaX=0)
    return output


def _generate_output_image(image_path: str = None, output: np.ndarray = None, cfg: DictConfig = None, input_size: list = None, output_details: dict = None):
    """
    Post-processing to convert raw output to segmentation output and then display input image with segmentation overlay

    Args:
        image_path (str): path to the network input image
        output (np.ndarray): corresponding network output
        cfg (dict): A dictionary containing the entire configuration file.
        input_size (list): [height, width] of the input image

    Returns:
        None
    """

    # directory for saving prediction outputs
    prediction_result_dir = f'{cfg.output_dir}/predictions/'
    os.makedirs(prediction_result_dir, exist_ok=True)

    # Load the original image
    original_image = tf.io.read_file(image_path)
    original_image = tf.image.decode_image(original_image, channels=3)
    
    original_image = preprocess_image(original_image, height=input_size[0], width=input_size[1], aspect_ratio=cfg.preprocessing.resizing.aspect_ratio, 
                                      interpolation=cfg.preprocessing.resizing.interpolation, scale=None, offset=None, perform_scaling=False)
    original_image = original_image.numpy()
    original_image = original_image.astype(np.uint8)
    
    if not cfg.general.display_figures:
        plt.ioff()

    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[10, 10])

    # Plot input image
    plt.subplot(grid_spec[0])
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Original image')
    
    plt.subplot(grid_spec[1])
    plt.imshow(output, cmap="plasma")
    plt.axis('off')
    plt.title('Output image')

    # Save figure in the predictions directory
    fig_image_name = os.path.split(image_path)[1]
    pred_res_filename = f'{prediction_result_dir}/{os.path.basename(fig_image_name.split(".")[0])}.png'
    plt.savefig(pred_res_filename, bbox_inches='tight')

    if cfg.general.display_figures:
        plt.waitforbuttonpress()

    plt.close()

   
def predict(cfg: DictConfig = None) -> None:
    """
    Predicts a class for all the images that are inside a given directory.
    The model used for the predictions can be either a .h5 or .tflite file.

    Args:
        cfg (dict): A dictionary containing the entire configuration file.

    Returns:
        None

    Errors:
        The directory containing the images cannot be found.
        The directory does not contain any file.
        An image file can't be loaded.
    """

    model_path = cfg.general.model_path
    model_type = cfg.general.model_type
    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)
    test_images_dir = cfg.prediction.test_files_path
    cpp = cfg.preprocessing

    _, model_input_shape = get_model_name_and_its_input_shape(model_path)
    height, width = model_input_shape[1:] if Path(model_path).suffix == '.onnx' else model_input_shape[0:2]

    print("[INFO] Making predictions using:")
    print("  model:", model_path)
    print("  images directory:", test_images_dir)

    # Load the test images
    image_filenames = []
    images = []

    channels = 1 if cpp.color_mode == "grayscale" else 3
    aspect_ratio = cpp.resizing.aspect_ratio
    interpolation = cpp.resizing.interpolation
    scale = cpp.rescaling.scale
    offset = cpp.rescaling.offset

    for fn in os.listdir(test_images_dir):
        im_path = os.path.join(test_images_dir, fn)
        # Skip subdirectories if any
        if os.path.isdir(im_path):
            continue
        image_filenames.append(im_path)
        # Load the image
        try:
            data = tf.io.read_file(im_path)
            img = tf.image.decode_image(data, channels=channels)
        except:
            raise ValueError(f"\nUnable to load image file {im_path}\n"
                             "Supported image file formats are BMP, GIF, JPEG and PNG.")
        # Preprocess the image
        img = preprocess_image(img, height=height, width=width, aspect_ratio=aspect_ratio, interpolation=interpolation,
                               scale=scale, offset=offset, perform_scaling=True)
        images.append(img)

    if not images:
        raise ValueError("Unable to make predictions, could not find any image file in the "
                         f"images directory.\nReceived directory path {test_images_dir}")

    file_extension = Path(model_path).suffix
    if file_extension == ".h5":
        # Load the .h5 model
        model = tf.keras.models.load_model(model_path, compile=False)

        for i in range(len(images)):
            img = preprocess_input(images[i], input_details=None)
            raw_prediction = model.predict(img)
            output = postprocess_output_values(output=raw_prediction, output_details=None)

            # generation of output images
            output = np.squeeze(output)
            if model_type == "fast_depth":
                max_val = np.max(output)
                output = max_val - output
            _generate_output_image(image_path=image_filenames[i], output=output, cfg=cfg, input_size=[height, width], output_details=None)

    elif file_extension == ".tflite":
        # Load the Tflite model and allocate tensors
        interpreter_quant = tf.lite.Interpreter(model_path=model_path)
        interpreter_quant.allocate_tensors()
        input_details = interpreter_quant.get_input_details()[0]
        input_index_quant = input_details["index"]
        output_details = interpreter_quant.get_output_details()[0]
        output_index_quant = output_details["index"]

        ai_runner_interpreter = ai_runner_interp(target,name_model)

        for i in range(len(images)):
            if target == "host":
                img = preprocess_input(images[i], input_details=input_details)
                interpreter_quant.set_tensor(input_index_quant, img)
                interpreter_quant.invoke()
                raw_prediction = interpreter_quant.get_tensor(output_index_quant)
                output = postprocess_output_values(output=raw_prediction, output_details=output_details)
            elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                data   = ai_interp_input_quant(ai_runner_interpreter,images[i].numpy()[None],cpp.rescaling.scale,cpp.rescaling.offset,file_extension)
                output = ai_runner_invoke(data,ai_runner_interpreter)
                output = ai_interp_outputs_dequant(ai_runner_interpreter,output)[0]

            # generation of output images
            output = np.squeeze(output)
            if model_type == "fast_depth":
                max_val = np.max(output)
                output = max_val - output
            _generate_output_image(image_path=image_filenames[i], output=output, cfg=cfg, input_size=[height, width], output_details=output_details)

    elif file_extension == ".onnx":
        images = np.stack(images, axis=0)
        images = images.transpose((0, 3, 1, 2))
        sess = onnxruntime.InferenceSession(model_path)

        ai_runner_interpreter = ai_runner_interp(target,name_model)

        for i in range(len(images)):
            if target == "host":
                img = preprocess_input(images[i], input_details=None)
                img = img.numpy()
                raw_prediction = predict_onnx(sess, img)
                output = postprocess_output_values(output=raw_prediction, output_details=None)
                # we consider by default that output is channel first, so we transpose to have same shape as input image
                output = output.transpose(0, 2, 3, 1)
            elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                data   = ai_interp_input_quant(ai_runner_interpreter,images[i][None],cpp.rescaling.scale,cpp.rescaling.offset,file_extension)
                output = ai_runner_invoke(data,ai_runner_interpreter)
                output = ai_interp_outputs_dequant(ai_runner_interpreter,output)[0]

            # generation of output images
            output = output[0,:,:,0]
            if model_type == "fast_depth":
                max_val = np.max(output)
                output = max_val - output
            _generate_output_image(image_path=image_filenames[i], output=output, cfg=cfg, input_size=[height, width], output_details=None)

    else:
        raise TypeError(f"Unknown or unsupported model type. Received path {model_path}")
