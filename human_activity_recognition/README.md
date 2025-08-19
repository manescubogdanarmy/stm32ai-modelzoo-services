# Human Activity Recognition STM32 Model Zoo


## Directory components:
* [datasets](docs/README_DATASETS.md) placeholder for the human activity recognition datasets.
* [docs](docs/) contains all readmes and documentation specific to the human activity recognition use case.
* [src](docs/README_OVERVIEW.md) contains tools to train, evaluate, benchmark, quantize and deploy your model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be single operation or a set of chained operations.

You can refer to readme links below that provide typical examples of operation modes, and tutorials on specific services:

   - [training, chain_tb](./docs/README_TRAINING.md)
   - [evaluation](./docs/README_EVALUATION.md)
   - [benchmarking](./docs/README_BENCHMARKING.md)
   - [deployment](./docs/README_DEPLOYMENT.md)

All .yaml configuration examples are located in [config_file_examples](./src/config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `training`| Train an HAR |
| `evaluation` | Evaluate the accuracy of a float model on a test or validation dataset|
| `benchmarking` | Benchmark a float model on an STM32 board |
| `deployment`   | Deploy a model on an STM32 board |
| `chain_tb`  | Sequentially: training, benchmarking of trained model |


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How to define and train my own model?](./docs/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](./docs/tuto/how_to_deploy_a_model_on_a_target.md)

Remember that minimalistic yaml files are available [here](./src/config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

