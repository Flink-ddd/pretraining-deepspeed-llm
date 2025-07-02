
import argparse
import torch
import deepspeed
import time
import os
import torch.distributed as dist # 导入 torch.distributed

from src.models.transformer import Transformer

def get_dummy_data(batch_size, context_length, vocab_size, device):
    """生成用于Debug的虚拟数据"""
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y

def main():
    print("--- DeepSpeed Debug Script ---")

    parser = argparse.ArgumentParser(description="DeepSpeed Debugging")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 手动设置设备，这在调试单卡时非常重要
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # 手动初始化 PyTorch 分布式进程组
    # 这会使用我们在 launch.json 中设置的环境变量
    print("Initializing torch.distributed...")
    dist.init_process_group(backend='nccl')
    print("torch.distributed initialized.")
    # ----------------------------------------------------

    # 2. 定义一个极小的模型
    tiny_model_args = {
        'vocab_size': 50257,
        'context_length': 128,
        'n_embed': 128,
        'n_head': 4,
        'N_BLOCKS': 2,
    }
    model = Transformer(**tiny_model_args)
    
    # 3. DeepSpeed 初始化！
    # 现在它会检测到已经存在的分布式环境，并直接使用它
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
        xb, yb = get_dummy_data(4, 128, tiny_model_args['vocab_size'], device)

        print(f"\n--- Step {step} ---")
        print("Forward pass...")
        loss = model_engine(xb, yb)[1]
        
        print("Backward pass...")
        model_engine.backward(loss)
        
        print("Optimizer step...")
        model_engine.step()

        print(f"Step {step}: Loss = {loss.item()}")
        time.sleep(1) 

    print("\nDebug script finished successfully!")
    
    # 清理进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    main()