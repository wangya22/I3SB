#!/bin/bash
python sample.py \
--ckpt sr4x-bicubic \
--n-gpu-per-node 1 \
--dataset-dir /mnt/CAMCA/home/yuang/Imagenet \
--batch-size 1 \
--use-fp16 \
--clip-denoise \
--nfe 25 \
--use-i3sb \
--eta 0.6

