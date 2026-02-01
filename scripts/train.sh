# scripts/train.sh
#!/bin/bash

# 基本训练
python main/train.py \
    --data-path data/RBRTEd.csv \
    --model-type LSTM \
    --hidden-size 128 \
    --num-layers 3 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --save-dir checkpoints/lstm/ \
    --results-dir results/lstm/

# 使用配置文件训练
python main/train.py --config configs/default.yaml

# 继续训练
python main/train.py \
    --checkpoint-path checkpoints/lstm/best_model.pth \
    --epochs 50