# 大型语言模型训练与推理框架

本仓库提供了一套完整的大型语言模型（LLM）训练与推理框架，支持监督微调（SFT）、奖励模型训练（RM）、强化学习训练（RL）和各种推理解码策略。

## 项目结构

```
codebase/
├── src/                      # 源代码目录
│   ├── inference/            # 推理框架
│   │   ├── base.py           # 推理基类
│   │   ├── greedy/           # 贪婪解码
│   │   ├── beam_search/      # 集束搜索
│   │   ├── best-of-n/        # Best-of-N 采样
│   │   ├── mcts/             # 蒙特卡洛树搜索
│   │   └── self-improve/     # 自我改进
│   ├── initialization/       # 初始化训练
│   │   ├── supervised_finetune/  # 监督微调
│   │   └── reward_model_training/ # 奖励模型训练
│   ├── trainer/              # 强化学习训练
│   │   ├── training_manager/ # 训练管理器
│   │   ├── verl/             # VERL框架
│   │   └── algorithms/       # 算法实现
│   └── data/                 # 数据处理
├── examples/                 # 示例代码
│   ├── RewardSQL/            # SQL任务训练示例
│   ├── SQL_optimizer/        # SQL优化器示例
│   └── DeepData/             # 深度数据分析示例
└── README.md                 # 本文档
```

## 主要功能

### 推理框架（Inference）

提供灵活的推理接口，支持多种解码策略：

- **基础解码**：贪婪解码、集束搜索等
- **高级策略**：Best-of-N、MCTS（蒙特卡洛树搜索）等
- **自我改进**：利用多轮反馈进行生成结果优化

通过统一的接口设计，可以轻松扩展新的解码策略和模型。

### 初始化训练（Initialization）

包含模型初始训练所需的脚本和工具：

- **监督微调（SFT）**：使用标注数据对预训练模型进行微调
- **奖励模型训练（RM）**：训练用于评估生成质量的奖励模型

支持多种训练加速技术，如DeepSpeed、Flash Attention等。

### 强化学习训练（Trainer）

基于VERL框架的强化学习训练系统：

- **组件化设计**：可自定义奖励函数、算法实现等
- **算法支持**：PPO、GRPO、DAPO、Prime等
- **训练管理器**：提供简单易用的训练接口

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装VERL框架（可选，仅用于强化学习训练）
cd src/trainer/verl
pip install -e .
```

### 监督微调（SFT）

```bash
python src/initialization/supervised_finetune/sft.py \
    --config examples/RewardSQL/sft_config.json
```

### 奖励模型训练（RM）

```bash
python src/initialization/reward_model_training/rm_training.py \
    --config examples/RewardSQL/rm_config.json
```

### 强化学习训练（RL）

```bash
python examples/RewardSQL/train_verl.py \
    --config examples/RewardSQL/verl/grpo_trainer.yaml \
    --algorithm grpo \
    --data_path path/to/data \
    --output_dir outputs/verl_train
```

### 推理

```python
from src.inference import InferenceBase
from transformers import AutoModelForCausalLM, AutoTokenizer

# 创建推理实例
model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

inferencer = InferenceBase(model, tokenizer)

# 使用不同的解码策略
result = inferencer.generate(
    prompt="请帮我优化下面的SQL查询：SELECT * FROM users",
    decoding_strategy="greedy",  # 或 "beam", "best_of_n", "mcts" 等
    decoding_kwargs={"max_length": 1024, "temperature": 0.7}
)
print(result)
```

## 示例

本项目包含多个示例，展示如何在不同任务上使用本框架：

- **SQL任务**：在`examples/RewardSQL`目录下，演示如何训练和部署SQL优化和生成模型
- **自然语言处理**：使用不同的解码策略进行文本生成和优化

## 扩展指南

### 添加新的解码策略

1. 在`src/inference/`下创建新的策略目录和实现
2. 在`InferenceBase`类中注册新策略
3. 实现必要的接口方法

### 添加新的强化学习算法

1. 实现核心算法模块（优势函数等）
2. 注册算法到组件注册表
3. 在`VERLTrainingManager`中添加对新算法的支持

## 许可证

[MIT License](LICENSE) 