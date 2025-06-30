# config/config.py

# --- 数据集和分词器配置 ---
DATASET_NAME = "openwebtext"
TOKENIZER_NAME = "gpt2"

# --- 模型配置 (约 13B 参数) ---
# 这是在单卡 A40 上通过 DeepSpeed ZeRO-Offload 可以挑战的规模
VOCAB_SIZE = 50257
CONTEXT_LENGTH = 1024
N_EMBED = 5120  # 增加嵌入维度
N_HEAD = 40     # 增加头数
N_BLOCKS = 40   # 增加层数

# --- 训练配置 ---
# 注意：这现在是 'train_micro_batch_size_per_gpu'
# 总批量大小 (Global Batch Size) 将在 deepspeed config 中通过 gradient_accumulation_steps 控制
T_BATCH_SIZE = 4
T_TRAIN_STEPS = 500000
T_LR = 1e-5  # 为大模型调整学习率

# --- 评估和保存 ---
T_EVAL_STEPS = 250
T_EVAL_ITERS = 100
# 模型保存将由 DeepSpeed 接管，这里仅作参考
T_OUT_PATH = "models/ds_transformer_13B" 

# --- 设备配置 ---
DEVICE = 'cuda'
DTYPE = 'bfloat16'

# --- 文件路径 ---
TRAIN_PATH = f"data/{DATASET_NAME}_train.bin"
VAL_PATH = f"data/{DATASET_NAME}_val.bin"

# 将所有配置存入字典
default_config = {
    'dataset_name': DATASET_NAME,
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'val_path': VAL_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_train_steps': T_TRAIN_STEPS,
    't_lr': T_LR,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
    'dtype': DTYPE
}