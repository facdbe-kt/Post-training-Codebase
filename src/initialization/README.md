# 模型初始化训练

本目录包含用于模型初始化训练的脚本，主要包括监督微调（SFT）和奖励模型（RM）训练。

## 目录结构

```
initialization/
├── supervised_finetune/      # 监督微调相关代码
│   ├── sft.py                # 监督微调训练脚本
│   └── ms-swift/             # Swift微调工具
└── reward_model_training/    # 奖励模型训练相关代码
    ├── rm_training.py        # 奖励模型训练脚本
    └── OpenRLHF/             # OpenRLHF强化学习框架
```

## 监督微调（SFT）

`sft.py` 是一个灵活的监督微调训练脚本，支持通过命令行参数或配置文件设置超参数。

### 功能特点

- 支持加载预训练模型（本地路径或ModelScope/HuggingFace模型ID）
- 灵活的数据集配置与处理
- LoRA高效参数微调
- 集成DeepSpeed分布式训练
- 支持FP16/BF16精度训练
- 可通过命令行参数或配置文件设置训练参数

### 使用方法

**基本用法**:

```bash
python sft.py --model_id_or_path Qwen/Qwen2.5-7B-Instruct --output_dir ./output/sft-model
```

**使用配置文件**:

```bash
python sft.py --config path/to/config.json
```

**常用参数**:

- `--model_id_or_path`: 预训练模型ID或路径
- `--output_dir`: 输出目录
- `--dataset`: 训练数据集名称或路径
- `--learning_rate`: 学习率
- `--per_device_train_batch_size`: 每设备训练批大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--num_train_epochs`: 训练轮数
- `--lora_rank`: LoRA秩
- `--torch_dtype`: 模型精度（float32/float16/bfloat16）

### 配置文件示例

```json
{
  "model_id_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "output_dir": "output/sft-qwen",
  "dataset": ["AI-ModelScope/alpaca-gpt4-data-zh"],
  "learning_rate": 1e-4,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "num_train_epochs": 2,
  "lora_rank": 8,
  "torch_dtype": "bfloat16"
}
```

## 奖励模型（RM）训练

`rm_training.py` 是基于OpenRLHF框架的奖励模型训练脚本，支持多种设置和配置选项。

### 功能特点

- 基于OpenRLHF框架实现奖励模型训练
- 支持偏好数据集训练
- 集成DeepSpeed ZeRO优化
- 支持Flash Attention加速
- 配置灵活，支持命令行参数和配置文件

### 使用方法

**基本用法**:

```bash
python rm_training.py --model_id_or_path Qwen/Qwen2.5-7B-Instruct --output_dir ./output/rm-model --dataset path/to/preference/dataset
```

**使用配置文件**:

```bash
python rm_training.py --config path/to/rm_config.json
```

**常用参数**:

- `--model_id_or_path`: 预训练模型ID或路径
- `--output_dir`: 输出目录
- `--dataset`: 偏好数据集路径
- `--chosen_key`: 偏好数据集中选择文本的键名（默认: "chosen"）
- `--rejected_key`: 偏好数据集中拒绝文本的键名（默认: "rejected"）
- `--train_batch_size`: 训练批次大小
- `--learning_rate`: 学习率
- `--max_epochs`: 训练轮数
- `--torch_dtype`: 模型精度
- `--zero_stage`: DeepSpeed ZeRO优化阶段

### 配置文件示例

```json
{
  "model_id_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "output_dir": "output/rm-model",
  "dataset": "OpenRLHF/preference_dataset_mixture",
  "chosen_key": "chosen",
  "rejected_key": "rejected",
  "train_batch_size": 256,
  "micro_train_batch_size": 1,
  "learning_rate": 9e-6,
  "max_epochs": 1,
  "zero_stage": 3,
  "torch_dtype": "bfloat16",
  "flash_attn": true,
  "gradient_checkpointing": true
}
```