# scripts/generate_text.py
import torch
import tiktoken
import argparse
import deepspeed
from config.config import default_config as config
from src.models.transformer import Transformer

def generate_text(model_checkpoint_path: str, input_text: str, max_new_tokens: int = 100, device: str = 'cuda') -> str:
    """
    使用 DeepSpeed 训练的 Transformer 模型生成文本。
    """
    # 初始化模型结构
    model_args = {k: config[k] for k in ['vocab_size', 'n_embed', 'n_head', 'n_blocks', 'context_length']}
    model = Transformer(**model_args)

    # 加载 DeepSpeed 检查点
    # DeepSpeed 会自动处理参数的重新组合
    # 注意：load_path 是检查点目录，而不是单个文件
    _, client_sd = model.load_checkpoint(model_checkpoint_path)
    
    # client_sd 可能包含其他状态，我们只需要模型权重
    model.load_state_dict(client_sd['model_state_dict'])

    model.eval().to(device)
    print("模型加载成功！")

    # 分词器
    enc = tiktoken.get_encoding("gpt2")

    start_ids = enc.encode_ordinary(input_text)
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 生成过程
    print("正在生成文本...")
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()

    output_text = enc.decode(generated_tokens)
    return output_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained DeepSpeed Transformer model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the DeepSpeed model checkpoint directory.')
    parser.add_argument('--input_text', type=str, default="Hello, I'm a language model,", help='The initial text.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run generation on.')
    
    args = parser.parse_args()

    generated = generate_text(args.model_path, args.input_text, args.max_new_tokens, args.device)
    print("------------------")
    print(f"Generated text:\n{generated}")

if __name__ == "__main__":
    main()