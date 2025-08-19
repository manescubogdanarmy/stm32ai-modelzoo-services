# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils import model_family

              
def decode_ssd_predictions(predictions: tuple, clip_boxes: bool = True) -> tuple:
    """
    An SSD model outputs anchor boxes and offsets. This function
    applies the offsets to the anchor boxes to obtain the coordinates
    of the bounding boxes predicted by the model.
    
    Arguments:
        predictions:
            The SSD model output, a tuple of 3 elements:
                1. Scores of predicted boxes.
                   A tf.Tensor with shape [batch_size, num_anchors, num_classes]
                2. Offsets.
                   A tf.Tensor with shape [batch_size, num_anchors, 4]
                3. Anchor boxes.
                   A tf.Tensor with shape [batch_size, num_anchors, 4]
        clip_boxes:
            A boolean. If True, the coordinates of the output bounding boxes
            are clipped to fit the image. If False, they are left as is. 
            Defaults to True.

    Returns:
        scores:
            The scores of the predicted bounding boxes in each class.
            A tf.Tensor with shape [batch_size, num_anchors, num_classes]
        boxes:
            The predicted bounding boxes in the (x1, y1, x2, y2) coordinates
            system. (x1, y1) and (x2, y2) are pairs of diagonally opposite 
            corners. The coordinates values are normalized.
            A tf.Tensor with shape [batch_size, num_anchors, 4]
    """

    scores = predictions[0]
    raw_boxes = predictions[1]
    anchor_boxes = predictions[2]
    
    # Apply anchor offsets to the detection boxes
    x1 = raw_boxes[..., 0] * (anchor_boxes[..., 2] - anchor_boxes[..., 0]) + anchor_boxes[..., 0]
    x2 = raw_boxes[..., 2] * (anchor_boxes[..., 2] - anchor_boxes[..., 0]) + anchor_boxes[..., 2]
    y1 = raw_boxes[..., 1] * (anchor_boxes[..., 3] - anchor_boxes[..., 1]) + anchor_boxes[..., 1]
    y2 = raw_boxes[..., 3] * (anchor_boxes[..., 3] - anchor_boxes[..., 1]) + anchor_boxes[..., 3]

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    if clip_boxes:
        boxes = tf.clip_by_value(boxes, 0, 1)
       
    # Get rid of the background class
    scores = scores[..., 1:]
    
    return boxes, scores

    
def yolo_head(feats, anchors, num_classes):
    """
    Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """

    num_anchors = tf.shape(anchors)[0]
    anchors = tf.reshape(anchors, [1, 1, 1, num_anchors, 2])

    # Get the dimensions of the grid of cells
    conv_dims = tf.shape(feats)[1:3]
    
    # Generate the grid cell indices
    # Note: YOLO iterates over height index before width index.
    i = tf.where(tf.ones([conv_dims[1], conv_dims[0]], dtype=tf.bool))
    conv_index = tf.stack([i[:, 1], i[:, 0]], axis=-1)
    
    # The coordinates box_xy of the centers of prediction boxes
    # are relative to the top-left corner of the grid cells.
    feats = tf.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    box_xy = tf.math.sigmoid(feats[..., :2])
    box_wh = tf.math.exp(feats[..., 2:4])
    box_confidence = tf.math.sigmoid(feats[..., 4:5])
    box_class_probs = tf.math.softmax(feats[..., 5:])

    conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = tf.cast(conv_index, tf.float32)
    conv_dims = tf.reshape(conv_dims, [1, 1, 1, 1, 2])
    conv_dims = tf.cast(conv_dims, tf.float32)

    # Adjust the coordinates of the centers of prediction 
    # boxes to grid cell locations so that they are 
    # relative to the top-left corner of the image.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs

    
def decode_yolo_predictions(predictions, num_classes, anchors, image_size):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(predictions, anchors, num_classes)

    x = box_xy[..., 0]
    y = box_xy[..., 1]
    w = box_wh[..., 0]
    h = box_wh[..., 1]
    boxes = tf.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=-1)
    boxes = tf.clip_by_value(boxes, 0, 1)   
    
    # Flatten boxes using the shape of box_confidence which is 
    # [batch_size, yolo_grid_nrows, yolo_grid_ncols, num_anchors]
    conf_shape = tf.shape(box_confidence)
    batch_size = conf_shape[0]
    num_boxes = conf_shape[1] * conf_shape[2] * conf_shape[3]

    boxes = tf.reshape(boxes, [batch_size, num_boxes, 4])
    
    box_confidence = tf.reshape(box_confidence, [batch_size, num_boxes, 1])
    box_class_probs = tf.reshape(box_class_probs, [batch_size, num_boxes, num_classes])
    scores = box_confidence * box_class_probs
    
    return boxes, scores

def decode_yolo_v8_predictions(predictions):
    x = predictions[..., 0]
    y = predictions[..., 1]
    w = predictions[..., 2]
    h = predictions[..., 3]
    boxes = tf.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=-1)
    boxes = tf.clip_by_value(boxes, 0, 1)
    scores = predictions[..., 4:]
    return boxes, scores

def _ssd_generate_anchors(opts):
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)

def remove_kpts(arr):
    if arr.shape[1] > 1:
        return arr[:, :4]
    return arr

def sort_and_combine(arrays):
    squeezed_arrays = [np.squeeze(arr, axis=0) for arr in arrays]
    sorted_arrays = sorted(squeezed_arrays, key=lambda arr: (arr.shape[0], arr.shape[1]), reverse=True)
    processed_arrays = [remove_kpts(arr) for arr in sorted_arrays]
    out_1 = np.concatenate((processed_arrays[0], processed_arrays[1]), axis=1)
    out_2 = np.concatenate((processed_arrays[2], processed_arrays[3]), axis=1)
    final_array = np.concatenate((out_1, out_2), axis=0)
    return  final_array

def _decode_boxes(raw_boxes,input_shape,anchors):
    # width == height so scale is the same across the board
    scale = input_shape
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += anchors
    for i in range(2, num_points):
        boxes[:, i] += anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes

def sigmoid(data):
    return 1 / (1 + np.exp(-data))

def _get_sigmoid_scores(raw_scores):
    """Extracted loop from ProcessCPU (line 327) in
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
    # # this lower limit is safe for use with the sigmoid functions and float32
    RAW_SCORE_LIMIT = 80
    # just a single class ("face"), which simplifies this a lot
    # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
    #    causes overflow with IEEE single precision floats (max ~10e38)
    raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
    raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
    # 2) apply sigmoid function on clipped confidence scores
    return sigmoid(raw_scores)

def decode_face_detect_front_predictions(predictions,image_size):
    SSD_OPTIONS_FRONT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0}
    anchors=_ssd_generate_anchors(SSD_OPTIONS_FRONT)
    out = sort_and_combine(predictions)
    decoded_boxes = _decode_boxes(out[:,0:4],image_size[0],anchors) #xmin, ymin, xmax, ymax
    reshaped_d_boxes = decoded_boxes.reshape(-1, 4)
    decoded_scores = _get_sigmoid_scores(out[:,-1])
    reshaped_d_scores = np.expand_dims(decoded_scores, axis=1)
    boxes = np.expand_dims(reshaped_d_boxes, axis=0)
    scores = np.expand_dims(reshaped_d_scores, axis=0)
    return boxes, scores




def decode_yolo_v4_predictions(predictions):
    # Lists to hold respective values while unwrapping.
    boxes = np.clip(np.squeeze(predictions[0]), 0, 1)
    scores = np.squeeze(predictions[1])
    if(len(scores.shape)==1):
        scores = np.expand_dims(scores,axis=1)
    boxes = np.expand_dims(boxes, axis=0)
    scores = np.expand_dims(scores, axis=0)
    
    return boxes, scores

    

def nms_box_filtering(
                boxes: tf.Tensor,
                scores: tf.Tensor,
                max_boxes: int = None,
                score_threshold: float = None,
                iou_threshold: float = None,
                clip_boxes: bool = True) -> tuple:
    """
    Prunes detection boxes using non-max suppression (NMS).
    
    The coordinates of the input bounding boxes must be in the
    (x1, y1, x2, y2) system with normalized values. (x1, y1)
    and (x2, y2) are pairs of diagonally opposite corners.
    The output boxes are also in the (x1, y1, x2, y2) system 
    with normalized values.
    
    The NMS is class-aware, i.e. the IOU between two boxes assigned
    to different classes is 0.
    
    If the number of boxes selected by NMS is less than the maximum
    number, padding boxes with all 4 coordinates set to 0 are
    used to reach the maximum number.
    
    Arguments:
        boxes:
            Detection boxes to prune using NMS.
            A tf.Tensor with shape [batch_size, num_boxes, 4]
        scores:
            Scores of the detection boxes in each class.
            A tf.Tensor with shape [batch_size, num_boxes, num_classes]
        max_boxes:
            An integer, the maximum number of boxes to be selected
            by NMS.
        score_threshold:
            A float, the score threshold to use to discard 
            low-confidence boxes.
        iou_threshold:
            A float, the IOU threshold used to eliminate boxes that
            have a large overlap with a selected box.
        clip_boxes:
            A boolean. If True, the output coordinates of the boxes
            selected by NMS are clipped to fit the image. If False,
            they are left as is. Defaults to True.

    Returns:
        nmsed_boxes:
            Boxes selected by NMS. 
            A tf.Tensor with shape [batch_size, max_boxes, 4]
        nmsed_scores:
            Scores of the selected boxes.
            A tf.Tensor with shape [batch_size, max_boxes]
        nmsed_classes:
            Classes assigned to the selected boxes.
            A tf.Tensor with shape [batch_size, max_boxes]
    """
    
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    num_classes = tf.shape(scores)[-1]
    
    # Convert box coordinates from (x1, y1, x2, y2) to (y1, x1, y2, x2)
    boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

    # NMS is run by class, so we need to replicate the boxes num_classes times.
    boxes_t = tf.tile(boxes, [1, 1, num_classes])
    nms_input_boxes = tf.reshape(boxes_t, [batch_size, num_boxes, num_classes, 4])

    # The valid_detections output is not returned. Invalid boxes 
    # have all 4 coordinates set to 0, so they are easy to spot.    
    nmsed_boxes, nmsed_scores, nmsed_classes, _ = \
                tf.image.combined_non_max_suppression(
                        boxes=nms_input_boxes,
                        scores=scores,  
                        max_output_size_per_class=max_boxes,
                        max_total_size=max_boxes,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                        # Pad/clip output nmsed boxes, scores and classes to max_total_size
                        pad_per_class=False,
                        # Clip coordinates of output nmsed boxes to [0, 1]
                        clip_boxes=clip_boxes)

    # Convert coordinates of NMSed boxes to (x1, y1, x2, y2)
    nmsed_boxes = tf.stack([nmsed_boxes[..., 1], nmsed_boxes[..., 0],
                            nmsed_boxes[..., 3], nmsed_boxes[..., 2]],
                            axis=-1)

    return nmsed_boxes, nmsed_scores, nmsed_classes


def get_nmsed_detections(cfg, predictions, image_size):

    num_classes = len(cfg.dataset.class_names)
    cpp = cfg.postprocessing
    
    if model_family(cfg.general.model_type) == "ssd":
        boxes, scores = decode_ssd_predictions(predictions)
    elif model_family(cfg.general.model_type) == "yolo":
        boxes, scores = decode_yolo_predictions(predictions, num_classes, cpp.yolo_anchors, image_size)

    elif model_family(cfg.general.model_type) == "st_yolo_x":
        np_anchors=[]
        anchors = cpp.yolo_anchors
        network_stride = cpp.network_stride
        predictions = sorted(predictions, key=lambda x: x.shape[1], reverse=True)
        anchors = [anchors * (image_size[0]/ns) for ns in network_stride]
        for anch in anchors:
            if isinstance(anch, np.ndarray):
                np_anchors.append(anch.astype(np.float32))
            else:
                np_anchors.append(anch.numpy().astype(np.float32)) 
        levels_boxes = []
        levels_scores = []
        for i , prediction in enumerate(predictions):
            box, score = decode_yolo_predictions(prediction, num_classes, np_anchors[i], image_size)
            levels_boxes.append(box)
            levels_scores.append(score)
        
        boxes = tf.concat(levels_boxes, axis=1)
        scores = tf.concat(levels_scores, axis=1)
    
    elif model_family(cfg.general.model_type) == "yolo_v8":
        boxes, scores = decode_yolo_v8_predictions(predictions)
    elif model_family(cfg.general.model_type) == "yolo_v4":
        boxes, scores = decode_yolo_v4_predictions(predictions)
    elif model_family(cfg.general.model_type) == "face_detect_front":
        boxes, scores = decode_face_detect_front_predictions(predictions,image_size)

    else:
        raise ValueError("Unsupported model type")
        
    # NMS the detections
    nmsed_boxes, nmsed_scores, nmsed_classes = nms_box_filtering(
                    boxes,
                    scores,
                    max_boxes=cpp.max_detection_boxes,
                    score_threshold=cpp.confidence_thresh,
                    iou_threshold=cpp.NMS_thresh)

    return nmsed_boxes, nmsed_scores, nmsed_classes
    