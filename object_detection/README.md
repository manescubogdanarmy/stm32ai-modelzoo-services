# Object Detection STM32 Model Zoo

## Directory Components:
* [datasets](./docs/README_DATASETS.md) placeholder for the object detection datasets.
* [docs](./docs/) contains all readmes and documentation specific to the object detection use case.
* [src](./docs/README_OVERVIEW.md) contains tools to train, evaluate, benchmark, quantize and deploy your model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be a single operation or a set of chained operations.

You can refer to the README links below that provide typical examples of operation modes and tutorials on specific services:

- [training, chain_tqe, chain_tqeb](./docs/README_OVERVIEW.md)
- [quantization, chain_eqe, chain_qb](./docs/README_QUANTIZATION.md)
- [evaluation, chain_eqeb](./docs/README_EVALUATION.md)
- [benchmarking](./docs/README_BENCHMARKING.md)
- [prediction](./docs/README_PREDICTION.md)
- deployment, chain_qd ([STM32H7](./docs/README_DEPLOYMENT_STM32H7.md), [STM32N6](./docs/README_DEPLOYMENT_STM32N6.md), [STM32MPU](./docs/README_DEPLOYMENT_MPU.md))

All `.yaml` configuration examples are located in the [config_file_examples](./src/config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmarking, and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `training`| Train a model from the variety of object detection models in the model zoo **(BYOD)** or your own model with the same model type **(BYOM)** |
| `evaluation` | Evaluate the accuracy of a float or quantized model on a test or validation dataset|
| `quantization` | Quantize a float model |
| `prediction`   | Predict the classes some images belong to using a float or quantized model |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |
| `deployment`   | Deploy a model on an STM32 board |
| `chain_tqeb`  | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`    | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model |


The `model_type` attributes currently supported for the object detection are:
- `ssd_mobilenet_v2_fpnlite`: This is a Single Shot Detector (SSD) architecture that uses a MobileNetV2 backbone and a Feature Pyramid Network (FPN) head. It is designed to be fast and accurate, and is well-suited for use cases where real-time object detection is required.

- `st_ssd_mobilenet_v1 `: This is a variant of the SSD architecture that uses a MobileNetV1 backbone and a custom head(ST). It is designed to be robust to scale and orientation changes in the input images.

- `yolo_v8` : is an advanced object detection model from Ultralytics that builds upon the strengths of its predecessors in the YOLO series. It is designed for real-time object detection, offering high accuracy and speed. YOLOv8 incorporates state-of-the-art techniques such as improved backbone networks, better feature pyramid networks, and advanced anchor-free detection heads, making it highly efficient for various computer vision tasks.

- `yolo_v11` : is an advanced object detection model from Ultralytics that builds upon the strengths of its predecessors in the YOLO series. It is designed for real-time object detection, offering high accuracy and speed. YOLOv11 incorporates state-of-the-art techniques such as improved backbone networks, better feature pyramid networks, and advanced anchor-free detection heads, making it highly efficient for various computer vision tasks.

- `yolo_v5u`: (You Only Look Once version 5 from Ultralytics) is a popular object detection model known for its balance of speed and accuracy. It is part of the YOLO family and is designed to perform real-time object detection. 
 
- `st_yolo_x`: is an advanced object detection model that builds upon the YOLO (You Only Look Once) series, offering significant improvements in performance and flexibility. Unlike its predecessors, YOLOX can adopt an anchor-free approach, which simplifies the model and enhances its accuracy. It also incorporates advanced techniques such as decoupled head structures for classification and localization, and a more efficient training strategy. YOLOX is designed to achieve high accuracy and speed, making it suitable for real-time applications in various computer vision tasks. This ST variant embeds various tuning capabilities from the yaml configuration file.
 
- `st_yolo_lc_v1`: This is a lightweight version of the tiny yolo v2 object detection algorithm. It was optimized to work well on embedded devices with limited computational resources.

- `tiny_yolo_v2`: This is a lightweight version of the YOLO (You Only Look Once) object detection algorithm. It is designed to work well on embedded devices with limited computational resources.

- `yolo_v4`: YOLO (You Only Look Once) version 4 from Nvidia TAO (train adapt and optimize) toolkit with different possible backbones for feature extractions including different versions of resnet, mobilenet, darknet, efficientnet, vgg, and cspdarknet.

- `yolo_v4_tiny`: YOLO (You Only Look Once) version 4 __tiny__ from Nvidia TAO (train adapt and optimize) toolkit with backbones csp_darknet_tiny as the backbone.

- `face_detect_front` : BlazeFace Front 128x128 is a lightweight and efficient face detection model optimized for real-time applications on embedded devices. It is a variant of the BlazeFace architecture, designed specifically for detecting frontal faces at a resolution of 128x128 pixels.


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How can I use my own dataset?](./docs/tuto/how_to_use_my_own_object_detection_dataset.md)
* [How can I fine tune a pretrained model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](./docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I quantize, evaluate and deploy an Ultralytics Yolov8 model?](./docs/tuto/How_to_deploy_yolov8_yolov5_object_detection.md)
* [How can I evaluate my model on STM32N6 target?](./docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Remember that minimalistic yaml files are available [here](./src/config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

