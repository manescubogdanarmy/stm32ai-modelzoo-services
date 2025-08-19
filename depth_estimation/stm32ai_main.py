# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys
from hydra.core.hydra_config import HydraConfig
import hydra
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse
import logging
from typing import Optional
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from src.preprocessing import preprocess
from src.utils import get_config
from src.prediction import predict


# This function turns Tensorflow's eager mode on and off.
# Eager mode is for debugging the Model Zoo code and is slower.
# Do not set argument to True to avoid runtime penalties.
tf.config.run_functions_eagerly(False)




def process_mode(mode: str = None,
                 configs: DictConfig = None,
                 train_ds: tf.data.Dataset = None,
                 valid_ds: tf.data.Dataset = None,
                 quantization_ds: tf.data.Dataset = None,
                 test_ds: tf.data.Dataset = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        mode (str): The selected mode of operation. Must be one of 'train', 'evaluate', or 'predict'.
        configs (DictConfig): The configuration object.
        train_ds (tf.data.Dataset): The training dataset. Required if mode is 'train'.
        valid_ds (tf.data.Dataset): The validation dataset. Required if mode is 'train' or 'evaluate'.
        test_ds (tf.data.Dataset): The test dataset. Required if mode is 'evaluate' or 'predict'.
        quantization_ds(tf.data.Dataset): The quantization dataset.
    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """

    mlflow.log_param("model_path", configs.general.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    # Check the selected mode and perform the corresponding operation
   
    if mode == 'prediction':
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val_predict(cfg=configs)
        # Launches prediction on the target through the model zoo prediction service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predict(cfg=configs)
        print('[INFO] : Prediction complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs)
        print('[INFO] : Benchmark complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(configs.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stm32ai_version", configs.tools.stm32ai.version)
        mlflow.log_param("target", configs.benchmarking.board)
    # logging the completion of the chain
    log_to_file(configs.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(configs)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='depthest_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Extract the mode from the command-line arguments
    mode = cfg.operation_mode
    preprocess_output = preprocess(cfg=cfg)
    train_ds, valid_ds, quantization_ds, test_ds = preprocess_output
    # Process the selected mode
    process_mode(mode=mode, configs=cfg, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds,
                 test_ds=test_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
