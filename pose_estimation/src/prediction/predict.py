# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import onnxruntime
from hydra.core.hydra_config import HydraConfig

from common.utils import get_model_name_and_its_input_shape, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from src.utils import ai_runner_invoke, skeleton_connections_dict, stm32_to_opencv_colors_dict
from src.postprocessing  import spe_postprocess, heatmaps_spe_postprocess, yolo_mpe_postprocess, hand_landmarks_postprocess, head_landmarks_postprocess


def _load_test_data(directory: str):
    """
    Parse the training data and return a list of paths to annotation files.
    
    Args:
    - directory: A string representing the path to test set directory.
    
    Returns:
    - A list of strings representing the paths to test images.
    """
    annotation_lines = []
    path = directory+'/'
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            new_path = path+file
            annotation_lines.append(new_path)
    return annotation_lines

def predict(cfg):
    """
    Run inference on all the images within the test set.

    Args:
        cfg (config): The configuration file.
    Returns:
        None.
    """

    output_dir = HydraConfig.get().runtime.output_dir
    model_path = cfg.general.model_path
    model_type = cfg.general.model_type
    class_name = cfg.dataset.class_names[0] if cfg.dataset.class_names is not None else 'None'

    skeleton_connections = []
    
    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"

    _, input_shape = get_model_name_and_its_input_shape(model_path)

    name_model = os.path.basename(model_path)
    
    # Keras model
    if Path(model_path).suffix == '.h5':
        best_model = tf.keras.models.load_model(model_path)
    # TFlite model
    elif Path(model_path).suffix == '.tflite':
        print(f"Loading tflite runtime for inference of {name_model}")
        interpreter_quant = tf.lite.Interpreter(model_path=model_path)
        interpreter_quant.allocate_tensors()
        input_details = interpreter_quant.get_input_details()[0]
        outputs_details = interpreter_quant.get_output_details()
        input_index_quant = interpreter_quant.get_input_details()[0]["index"]
    elif Path(model_path).suffix == '.onnx':
        print(f"Loading onnx runtime for inference of {name_model}")
        input_shape = input_shape[1:]
        sess = onnxruntime.InferenceSession(model_path)
        inputs  = sess.get_inputs()
        outputs = sess.get_outputs()
    
    ai_runner_interpreter = ai_runner_interp(target,name_model)

    interpolation = cfg.preprocessing.resizing.interpolation
    if interpolation == 'bilinear':
        interpolation_type = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        interpolation_type = cv2.INTER_NEAREST
    else:
        raise ValueError("Invalid interpolation method. Supported methods are 'bilinear' and 'nearest'.")

    if cfg.prediction.test_files_path:
        test_set_path = cfg.prediction.test_files_path
        test_annotations =  _load_test_data(test_set_path)
    else:
        print("no test set found")

    prediction_result_dir = f'{cfg.output_dir}/predictions/'
    os.makedirs(prediction_result_dir, exist_ok=True)

    for img_id, image_file in enumerate(test_annotations):
        if image_file.endswith(".jpg") or file.endswith(".png"):
            
            print('Inference on image : ',image_file)

#            image = cv2.imread(os.path.join(test_set_path, image_file))
            image = cv2.imread(image_file)
            if len(image.shape) != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            img_name = os.path.splitext(image_file)[0]
            resized_image = cv2.resize(image, (int(input_shape[0]), int(input_shape[0])), interpolation=interpolation_type)
            image_processed = resized_image * cfg.preprocessing.rescaling.scale + cfg.preprocessing.rescaling.offset
            input_image_shape = [height, width]

            if Path(model_path).suffix == '.tflite' and target == 'host':

                if input_details['dtype'] == np.uint8:
                    image_processed = (image_processed - cfg.preprocessing.rescaling.offset) / cfg.preprocessing.rescaling.scale
                    image_processed = np.clip(np.round(image_processed), np.iinfo(input_details['dtype']).min, np.iinfo(input_details['dtype']).max)
                elif input_details['dtype'] == np.int8:
                    image_processed = (image_processed - cfg.preprocessing.rescaling.offset) / cfg.preprocessing.rescaling.scale
                    image_processed -= 128
                    image_processed = np.clip(np.round(image_processed), np.iinfo(input_details['dtype']).min, np.iinfo(input_details['dtype']).max)
                elif input_details['dtype'] == np.float32:
                    image_processed = image_processed
                else:
                    print('[ERROR] : input dtype not recognized -> ',input_details['dtype'])

                image_processed = image_processed.astype(input_details['dtype'])

            image_processed = np.expand_dims(image_processed, 0)

            if Path(model_path).suffix == '.h5':
                predictions = best_model.predict_on_batch(image_processed)

            elif Path(model_path).suffix == '.tflite':
                if target == 'host':
                    interpreter_quant.set_tensor(input_index_quant, image_processed)
                    interpreter_quant.invoke()
                    predictions = [interpreter_quant.get_tensor(outputs_details[j]["index"]) for j in range(len(outputs_details))]
                    # Add the support for quantized outputs, with retreival of scale & zero_point
                elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                    data        = ai_interp_input_quant(ai_runner_interpreter,image_processed,cfg.preprocessing.rescaling.scale, cfg.preprocessing.rescaling.offset,'.tflite')
                    predictions = ai_runner_invoke(data,ai_runner_interpreter)
                    predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

                # Dequantize the outputs with retreival of scale & zero_point
                for idx,pr in enumerate(predictions):
                    if outputs_details[idx]['dtype'] in [np.uint8,np.int8]:
                        predictions[idx] = (predictions[idx] - outputs_details[idx]['quantization'][1]) * outputs_details[idx]['quantization'][0]
                    elif outputs_details[idx]['dtype'] == np.float32:
                        pass
                    else:
                        print('[ERROR] : output dtype not recognized -> ',outputs_details[idx]['dtype'])
            elif Path(model_path).suffix == '.onnx':
                image_processed = np.transpose(image_processed,[0,3,1,2])
                if target == 'host':
                    predictions = sess.run([o.name for o in outputs], {inputs[0].name: image_processed.astype('float32')})
                elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                    data        = ai_interp_input_quant(ai_runner_interpreter,image_processed,cfg.preprocessing.rescaling.scale, cfg.preprocessing.rescaling.offset,'.onnx')
                    predictions = ai_runner_invoke(data,ai_runner_interpreter)
                    predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)
                for indp in range(len(predictions)):
                    if len(predictions[indp].shape)==3:
                        predictions[indp] = tf.transpose(predictions[indp],[0,2,1])
                    elif len(predictions[indp].shape)==4:
                        predictions[indp] = tf.transpose(predictions[indp],[0,2,3,1])

            if Path(model_path).suffix == '.h5':
                predictions_tensor = predictions
            elif Path(model_path).suffix in ['.tflite','.onnx']:
                if len(predictions)==1:
                    predictions_tensor = predictions[0]
                else:
                    predictions_tensor = predictions

            if model_type=='heatmaps_spe':
                poses = heatmaps_spe_postprocess(predictions_tensor)[0]
            elif model_type=='spe':
                poses = spe_postprocess(predictions_tensor)[0]
            elif model_type=='yolo_mpe':
                poses = yolo_mpe_postprocess(predictions_tensor,
                                             max_output_size = cfg.postprocessing.max_detection_boxes,
                                             iou_threshold   = cfg.postprocessing.NMS_thresh,
                                             score_threshold = cfg.postprocessing.confidence_thresh)[0]
            elif model_type=='hand_spe':
                poses,norm_poses,htype,hprob = hand_landmarks_postprocess(predictions_tensor)
            elif model_type=='head_spe':
                poses = head_landmarks_postprocess(predictions_tensor)
            else:
                print('No post-processing found for the model type : '+model_type)

            threshSkeleton = cfg.postprocessing.kpts_conf_thresh

            bbox_thick = int(0.6 * (height + width) / 600)

            radius = 3

            for ids,p in enumerate(poses):
                if model_type in ['heatmaps_spe','spe']:
                    xx, yy, pp = p[0::3],p[1::3],p[2::3]
                elif model_type == 'hand_spe':
                    xx = p[0::3]/input_shape[0]
                    yy = p[1::3]/input_shape[0]
                    pp = tf.ones_like(xx) * hprob[ids]
                    x1 = int(np.min(xx)*width)
                    x2 = int(np.max(xx)*width)
                    y1 = int(np.min(yy)*height)
                    y2 = int(np.max(yy)*height)
                elif model_type == 'head_spe':
                    radius = 1
                    xx = p[0::2]/input_shape[0]
                    yy = p[1::2]/input_shape[0]
                    pp = tf.ones_like(xx)
                    x1 = int(np.min(xx)*width)
                    x2 = int(np.max(xx)*width)
                    y1 = int(np.min(yy)*height)
                    y2 = int(np.max(yy)*height)
                elif model_type == 'yolo_mpe':
                    x,y,w,h,conf = p[:5]
                    xx, yy, pp = p[5::3],p[5+1::3],p[5+2::3]
                    if Path(model_path).suffix == '.onnx':
                        x  /= input_shape[0]
                        y  /= input_shape[0]
                        w  /= input_shape[0]
                        h  /= input_shape[0]
                        xx /= input_shape[0]
                        yy /= input_shape[0]
                    x1 = int((x - w/2)*width)
                    x2 = int((x + w/2)*width)
                    y1 = int((y - h/2)*height)
                    y2 = int((y + h/2)*height)
                
                if img_id==0:

                    kpts_nbr = cfg.dataset.keypoints if (cfg.dataset.keypoints is not None and cfg.dataset.keypoints==len(xx)) else len(xx)

                    try:
                        skeleton_connections = skeleton_connections_dict[class_name][kpts_nbr]
                    except:
                        list_of_connections=''
                        for s in skeleton_connections_dict : list_of_connections += s+": "+str(list(skeleton_connections_dict[s].keys()))+" | "
                        print(f'Skeleton for the class [{class_name}] & number of keypoints [{kpts_nbr}] is unknown -> use {list_of_connections[:-2]}')
                        print('You can add your own in the utils/connections.py file')

                if not tf.reduce_all(tf.constant(pp)==0):
                    for i in range(0,len(xx)):
                        if float(pp[i])>threshSkeleton:
                            cv2.circle(image,(int(xx[i]*width),int(yy[i]*height)),radius=radius,color=(255, 255, 255), thickness=-1)
                        else:
                            cv2.circle(image,(int(xx[i]*width),int(yy[i]*height)),radius=radius,color=(255, 0, 0), thickness=-1)
                    for k,l,color in skeleton_connections:
                        if float(pp[k])>threshSkeleton and float(pp[l])>threshSkeleton: 
                            cv2.line(image,(int(xx[k]*width),int(yy[k]*height)),(int(xx[l]*width),int(yy[l]*height)),stm32_to_opencv_colors_dict[color])

                if model_type=='yolo_mpe':
                    btext = '{}-{:.2f}'.format(class_name,conf)
                elif model_type=='hand_spe':
                    btext = '{}'.format(['left_hand','right_hand'][htype[ids]>0.5])

                if model_type in ['yolo_mpe','hand_spe']:
                    cv2.rectangle(image,(x1,y1), (x2, y2),(255, 0, 255),1)
                    cv2.putText(image, btext, (x1,y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), bbox_thick//2, lineType=cv2.LINE_AA)
                    
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # writing prediction result to the output dir
            pred_res_filename = f'{prediction_result_dir}/{os.path.basename(img_name)}.png'
            cv2.imwrite(pred_res_filename,image)
            if cfg.general.display_figures:
                cv2.imshow('image',image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()