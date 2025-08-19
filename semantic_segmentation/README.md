# Semantic segmentation STM32 model zoo


## Directory components:
* [datasets](./docs/README_DATASETS.md) placeholder for the semantic segmentation datasets.
* [docs](./docs/) contains all readmes and documentation specific to the semantic segmentation use case.
* [src](./docs/README_OVERVIEW.md) contains tools to train, evaluate, benchmark, quantize and deploy your model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be single operation or a set of chained operations.

You can refer to readme links below that provide typical examples of operation modes, and tutorials on specific services:

   - [training, chain_tqe (train + quantize + evaluate + benchmark), chain_tqeb](./docs/README_TRAINING.md)
   - [quantization, chain_eqe, chain_qb](./docs/README_QUANTIZATION.md)
   - [evaluation, chain_eqeb](./docs/README_EVALUATION.md)
   - [benchmarking](./docs/README_BENCHMARKING.md)
   - [prediction](./docs/README_PREDICTION.md)
   - deployment, chain_qd ([STM32N6](./docs/README_DEPLOYMENT_STM32N6.md), [STM32MPU](./docs/README_DEPLOYMENT_MPU.md))

All .yaml configuration examples are located in [config_file_examples](./src/config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations                                                                                                                                           |
|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `training`| Train a model from the variety of segmentation models in the model zoo **(BYOD)** or your own model **(BYOM)**                                       |
| `evaluation` | Evaluate the accuracy of a float or quantized model on a test or validation dataset                                                                  |
| `quantization` | Quantize a float model                                                                                                                               |
| `prediction`   | Predict the classes some images belong to using a float or quantized model                                                                           |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board                                                                                               |
| `deployment`   | Deploy a model on an STM32 board                                                                                                                     |
| `chain_tqeb`  | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`    | Sequentially: training, quantization of trained model, evaluation of quantized model                                                                 |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model                                                          |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model                                                                         |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model                             |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model                                                                           |

The `model_type` attributes currently supported for the semantic segmentation are:
- `deeplab_v3` : specified in "Rethinking Atrous Convolution for Semantic Image Segmentation" paper by Google.
It is composed of a backbone (encoder) that can be a Mobilenet V2 (width parameter alpha) or a ResNet-50 or 101 for example followed by an ASPP (Atrous Spatial Pyramid Pooling).


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How to define and train my own model?](./docs/tuto/how_to_define_and_train_my_own_model.md)
* [How can I fine tune a pretrained model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](./docs/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I evaluate my model on STM32N6 target?](./docs/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Remember that minimalistic yaml files are available [here](./src/config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

