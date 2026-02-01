# scripts/test.sh
#!/bin/bash

# 测试模型
python main/test.py \
    --model-path checkpoints/lstm/best_model.pth \
    --results-dir results/lstm/test/

# 使用不同参数测试
python main/test.py \
    --model-path checkpoints/lstm/best_model.pth \
    --sequence-length 20 \
    --batch-size 1