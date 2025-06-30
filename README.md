# pretraining-deepspeed-llm
based on transformer+deepspeed from 0 to 1 pretraining LLM 




配置文件与场景二（单机多卡:ds_config_multi_gpu.json）完全相同。区别仅在于启动方式。


运行 scripts/data_download.py 和 scripts/data_preprocess.py


启动训练:


单机单卡:
deepspeed --num_gpus=1 scripts/train_transformer.py --deepspeed --deepspeed_config ds_config_single_gpu.json


单机多卡 (例如8卡):
deepspeed --num_gpus=8 scripts/train_transformer.py --deepspeed --deepspeed_config ds_config_multi_gpu.json


多机多卡 (例如4台机器，每台8卡):
创建 hostfile: 创建一个名为 hostfile 的文本文件，内容是集群中每台机器的IP地址或主机名，以及每台机器上有多少张GPU。
192.168.1.1 slots=8
192.168.1.2 slots=8
192.168.1.3 slots=8
192.168.1.4 slots=8
然后在主节点上运行：
deepspeed --hostfile hostfile scripts/train_transformer.py --deepspeed --deepspeed_config ds_config_multi_gpu.json
