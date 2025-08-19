#!/bin/bash

stedgeai generate --model yolov8n_256_quant_pc_ii_seg_coco-st.tflite --target stm32n6 --st-neural-art default@user_neuralart_STM32N6570-DK.json --input-data-type uint8 --output-data-type int8
cp st_ai_output/network.c STM32N6570-DK/
cp st_ai_output/network_ecblobs.h STM32N6570-DK/
cp st_ai_output/network_atonbuf.xSPI2.raw STM32N6570-DK/network_data.xSPI2.bin
arm-none-eabi-objcopy -I binary STM32N6570-DK/network_data.xSPI2.bin --change-addresses 0x70380000 -O ihex STM32N6570-DK/network_data.hex
