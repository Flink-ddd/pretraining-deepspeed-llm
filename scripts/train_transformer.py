# scripts/train_transformer.py

import os
import time
import argparse
import torch
import deepspeed

from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator

class Trainer:
    def __init__(self, model_engine, train_iterator, val_iterator, device):
        self.model_engine = model_engine
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.device = device

    @torch.no_grad()
    def _evaluate(self):
        """评估模型"""
        self.model_engine.eval()
        losses = torch.zeros(config['t_eval_iters'])
        for k in range(config['t_eval_iters']):
            X, Y = next(self.val_iterator)
            X, Y = X.to(self.device), Y.to(self.device)
            _, loss = self.model_engine(X, Y)
            losses[k] = loss.item()
        self.model_engine.train()
        return losses.mean()

    def train(self):
        """训练循环"""
        self.model_engine.train()
        t0 = time.time()
        
        for step in range(config['t_train_steps']):
            # 获取数据批次
            xb, yb = next(self.train_iterator)
            xb, yb = xb.to(self.device), yb.to(self.device)

            # 前向传播 + 反向传播
            loss = self.model_engine(xb, yb)[1]
            self.model_engine.backward(loss)

            # 更新权重
            self.model_engine.step()

            # 日志记录和评估
            if step % config['t_eval_steps'] == 0 and self.model_engine.local_rank == 0:
                t1 = time.time()
                dt = t1 - t0
                val_loss = self._evaluate()
                print(f"Step {step}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, Time={dt*1000:.2f}ms")
                t0 = t1
        
        if self.model_engine.local_rank == 0:
            print(f"训练完成. 正在保存检查点到 {config['t_out_path']}")
            # 使用deepspeed保存检查点
            self.model_engine.save_checkpoint(config['t_out_path'])


def main():
    # --- DeepSpeed 参数解析 ---
    parser = argparse.ArgumentParser(description="DeepSpeed Transformer Training")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- 模型初始化 ---
    model_args = {k: config[k] for k in ['vocab_size', 'n_embed', 'n_head', 'n_blocks', 'context_length']}
    model = Transformer(**model_args)
    
    # --- 数据加载器 ---
    train_iterator = get_batch_iterator(
        config['train_path'], config['t_batch_size'], config['context_length']
    )
    val_iterator = get_batch_iterator(
        config['val_path'], config['t_batch_size'], config['context_length']
    )

    # --- DeepSpeed 初始化 ---
    # DeepSpeed 会自动处理模型、优化器、数据加载器和设备放置
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    device = model_engine.device

    # --- 训练 ---
    trainer = Trainer(model_engine, train_iterator, val_iterator, device)
    trainer.train()

if __name__ == '__main__':
    main()