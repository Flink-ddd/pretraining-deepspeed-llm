# scripts/debug_deepspeed.py

import argparse
import torch
import deepspeed
import time

# 注意：为了Debug，我们不再从config文件读取，而是在这里直接定义一个超小的模型
from src.models.transformer import Transformer

def get_dummy_data(batch_size, context_length, vocab_size, device):
    """生成用于Debug的虚拟数据"""
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y

def main():
    print("--- DeepSpeed Debug Script ---")

    # 1. DeepSpeed 需要的参数解析器
    parser = argparse.ArgumentParser(description="DeepSpeed Debugging")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 2. 定义一个极小的模型，用于快速启动和调试
    # 这可以让你迅速进入断点，而不用等大模型加载半天
    tiny_model_args = {
        'vocab_size': 50257,
        'context_length': 128,
        'n_embed': 128,      # 非常小的嵌入维度
        'n_head': 4,         # 很少的头
        'n_blocks': 2,       # 只有2层
    }
    model = Transformer(**tiny_model_args)
    
    # 3. DeepSpeed 初始化！这是你 Debug 的核心入口
    # 在这里设置断点，可以进入 DeepSpeed 的源码
    print("Initializing DeepSpeed Engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    print("DeepSpeed Engine Initialized.")
    
    device = model_engine.device

    # 训练循环（只跑几步）
    for step in range(10):
        # 获取虚拟数据
        xb, yb = get_dummy_data(4, 128, tiny_model_args['vocab_size'], device)

        # 在这里设置断点，跟踪模型的前向和后向传播
        print(f"\n--- Step {step} ---")
        print("Forward pass...")
        loss = model_engine(xb, yb)[1]
        
        print("Backward pass...")
        model_engine.backward(loss)
        
        print("Optimizer step...")
        model_engine.step()

        print(f"Step {step}: Loss = {loss.item()}")
        time.sleep(1) # 暂停一下，方便观察

    print("\nDebug script finished successfully!")


if __name__ == '__main__':
    main()