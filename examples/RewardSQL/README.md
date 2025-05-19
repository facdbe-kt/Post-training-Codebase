# RewardSQL 训练配置

本目录包含用于SQL相关任务的各类训练配置和脚本，支持监督微调（SFT）、奖励模型（RM/PRM）训练以及基于VERL的强化学习训练。

## 目录结构

```
RewardSQL/
├── sft_config.json        # SFT训练配置文件
├── rm_config.json         # 奖励模型训练配置文件
├── train_verl.py          # VERL强化学习训练脚本
├── verl/                  # VERL自定义组件目录
│   ├── core_algos.py      # 自定义核心算法实现
│   ├── custom.py          # 自定义奖励管理器
│   ├── fsdp_workers.py    # 自定义工作器实现
│   └── grpo_trainer.yaml  # GRPO训练器配置
└── README.md              # 说明文档
```

## 训练配置

### 监督微调（SFT）配置

`sft_config.json` 包含了SFT训练的所有参数设置，主要参数包括：

- 模型：Qwen2.5-Coder-7B-Instruct
- 数据集：SQL相关拒绝采样数据
- 训练设置：批大小、学习率、梯度累积等
- 加速设置：DeepSpeed、FlashAttention等

### 奖励模型（PRM）配置

`rm_config.json` 包含了奖励模型训练的所有参数设置，主要参数包括：

- 模型：Qwen2.5-Coder-7B-Instruct
- 数据集：SQL相关偏好数据
- 训练设置：批大小、学习率等
- 优化选项：ZeRO-3、梯度检查点等
- Wandb记录等设置

### VERL强化学习配置

`verl/grpo_trainer.yaml` 包含了VERL强化学习训练的配置，主要包括：

- 数据配置：训练文件、批量大小等
- Actor-Rollout-Ref配置：模型路径、优化设置等
- 奖励模型配置：自定义奖励函数、验证器等
- 算法配置：GRPO、PPO等算法参数，包括：
  - 优势函数类型：`adv_estimator`，可选"gae"、"grpo"、"rloo"等
  - 自定义核心算法函数：`core_function`，指定要使用的优势计算函数
- 训练配置：轮数、保存频率等

## 使用方法

### 运行SFT训练

使用配置文件：

```bash
python ../../src/initialization/supervised_finetune/sft.py --config ./sft_config.json
```

### 运行奖励模型训练

使用配置文件：

```bash
python ../../src/initialization/reward_model_training/rm_training.py --config ./rm_config.json
```

### 运行VERL强化学习训练

使用训练管理器接口，一行代码即可启动VERL训练：

```bash
python train_verl.py --config verl/grpo_trainer.yaml --algorithm grpo --data_path path/to/data --output_dir outputs/verl_train
```

参数说明：
- `--config`：VERL配置文件路径
- `--algorithm`：使用的算法，支持grpo、ppo、prime等
- `--data_path`：训练数据路径
- `--output_dir`：输出目录路径
- `--core_algos`：可选，自定义核心算法模块路径，默认使用examples.RewardSQL.verl.core_algos
- `--core_function`：可选，指定要使用的核心算法函数名称，如compute_weighted_grpo_advantage
- `--epochs`：可选，训练轮数，默认使用配置文件中的设置

### 自定义核心算法函数

本目录下的`verl/core_algos.py`实现了多种自定义优势计算函数：

1. `compute_grpo_process_advantage_avg_unique`：基于过程监督的GRPO优势函数，使用每个response中不重复分数的平均值作为归一化基准
2. `compute_weighted_grpo_advantage`：带权重的GRPO优势函数，平衡批次内样本的影响
3. `compute_token_wise_advantage`：token级别的优势函数，考虑每个token位置的分布

您可以通过配置文件中的`algorithm.core_function`字段指定要使用的函数：

```yaml
algorithm:
  adv_estimator: "grpo"
  core_function: "compute_weighted_grpo_advantage"  # 使用带权重的GRPO优势函数
  # 其他配置...
```

或者通过命令行参数指定：

```bash
python train_verl.py --config verl/grpo_trainer.yaml --algorithm grpo --data_path path/to/data --core_function compute_token_wise_advantage
```

如果既不在配置文件中指定，也不在命令行中指定，系统会自动搜索适合当前算法类型的函数。

## 自定义组件

### 自定义奖励函数

可以通过`@register_reward_function`装饰器注册自定义奖励函数：

```python
from src.trainer.training_manager import register_reward_function

@register_reward_function("sql_reward")
def sql_reward_function(port, db_id, question_id, solution_str, ground_truth, extra_info):
    # 实现SQL验证逻辑
    return score
```

### 自定义奖励管理器

可以通过`@register_reward_manager`装饰器注册自定义奖励管理器：

```python
from src.trainer.training_manager import register_reward_manager
from verl import DataProto

@register_reward_manager("sql_manager")
class SQLRewardManager:
    def __init__(self, tokenizer, num_examine, config, compute_score=None):
        self.tokenizer = tokenizer
        self.config = config
        self.compute_score = compute_score
    
    def __call__(self, data: DataProto):
        # 实现奖励计算逻辑
        return rewards
```

## 注意事项

1. 脚本中使用的是相对路径，请确保在正确目录下运行脚本。
2. 配置文件中包含的一些参数可能需要在实际使用时调整。
3. 训练前请确保数据路径、输出路径等设置正确且有足够的权限。
4. VERL训练需要配置GPU资源，确保您有足够的GPU内存。
5. 使用自定义组件时，确保组件与VERL框架兼容。 