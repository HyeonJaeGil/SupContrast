#! /bin/bash

python main_supcon.py \
--model resnet50 \
--batch_size 64 \
--num_workers 4 \
--size 480 640 \
--learning_rate 0.5 \
--temp 0.1 \
--cosine \
--method SimCLR \
--dataset path \
--data_folder /root/STHEREO-raw \
--mean  0.5438,0.5438,0.5438 \
--std   0.2104,0.2104,0.2104 \
--torchvision