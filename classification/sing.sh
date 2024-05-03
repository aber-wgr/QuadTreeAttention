#!/bin/bash

echo "running script"
source /.venv/bin/activate
python3.11 main.py --config configs/quadtree/f3.py --batch-size 4 --data-set OMIDB --data-path /scratch/a.wgr/OPTIMAM_NEW/png_images/casewise/896 --input-size 896 --epochs 30 --num_workers 1 --loss-type WeightedCrossEntropy --lr 0.001 --smoothing 0.0 --drop_path 0.0 --drop 0.0 --mixup 0.0 --cutmix 0.0 --weight-decay 0.0 --color-jitter 0.0 --reprob 0.0 --channels 1 --aa augmix-m2-w3-d2 --wandb-project QTA_OMIDB_SCW --seed 29 --no-repeated-aug

