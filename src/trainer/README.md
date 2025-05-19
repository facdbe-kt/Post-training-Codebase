# 训练管理器

本目录包含模型训练相关的代码，提供通用的训练接口和强化学习算法实现。

## 目录结构

```
trainer/
├── training_manager/        # 训练管理器模块
│   ├── __init__.py          # 模块初始化和导出
│   ├── base.py              # 基础训练管理器接口
│   ├── rl_training_manager.py # 强化学习训练管理器
│   ├── verl_training_manager.py # VERL训练管理器
│   └── component_registry.py  # 组件注册器
├── verl/                    # VERL框架
│   ├── verl/                # VERL核心代码
│   │   ├── trainer/         # 训练器实现
│   │   ├── models/          # 模型实现
│   │   ├── workers/         # 工作器实现
│   │   └── ...
│   ├── recipe/              # 算法实现
│   │   ├── prime/           # Prime算法
│   │   ├── dapo/            # DAPO算法
│   │   ├── drgrpo/          # DRGRPO算法
│   │   └── ...
│   └── ...
└── README.md                # 本文档
```

## 功能概述

本目录提供了以下主要功能：

1. **通用训练管理接口**：定义了统一的训练、评估、保存和加载接口
2. **强化学习训练支持**：支持PPO、GRPO、DAPO、Prime等算法
3. **组件注册机制**：通过装饰器注册和使用自定义组件
4. **VERL框架封装**：简化VERL框架的使用，提供更简洁的API

## 使用方法

### 基本使用

最简单的使用方式是通过`VERLTrainingManager`类：

```python
from src.trainer.training_manager import VERLTrainingManager

# 初始化训练管理器
manager = VERLTrainingManager(
    config="path/to/config.yaml",  # 配置文件路径
    algorithm_type="grpo",         # 算法类型
)

# 执行训练
stats = manager.train(
    data_path="path/to/data",      # 训练数据路径
    output_dir="path/to/output"    # 输出目录
)

# 保存模型
manager.save("path/to/save")
```

### 自定义组件

通过组件注册机制，可以方便地使用自定义组件：

```python
from src.trainer.training_manager import (
    register_reward_manager, 
    register_reward_function
)

# 注册自定义奖励管理器
@register_reward_manager("my_reward_manager")
class MyRewardManager:
    def __init__(self, tokenizer, num_examine, config, compute_score=None):
        self.tokenizer = tokenizer
        self.config = config
        self.compute_score = compute_score
    
    def __call__(self, data):
        # 自定义奖励计算逻辑
        pass

# 注册自定义奖励函数
@register_reward_function("my_reward_function")
def my_reward_function(input_text, response_text, **kwargs):
    # 自定义奖励计算逻辑
    return reward_score

# 在训练管理器中使用
from src.trainer.training_manager import registry

manager = VERLTrainingManager(
    config="path/to/config.yaml",
    algorithm_type="grpo",
    reward_function=registry.get_reward_function("my_reward_function"),
    reward_manager_class=registry.get_reward_manager("my_reward_manager")
)
```

### 自定义核心算法

您可以通过创建自定义算法模块来实现新的训练算法或优化现有算法：

```python
# 在custom_core_algos.py中实现自定义算法
import torch
from collections import defaultdict

def compute_custom_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    """自定义优势函数实现"""
    # 实现您的优势计算逻辑
    return advantages, returns

# 在训练管理器中使用
manager = VERLTrainingManager(
    config="path/to/config.yaml",
    algorithm_type="grpo",
    core_algos_module="path/to/custom_core_algos.py",
    core_algos_function="compute_custom_advantage"  # 指定要使用的函数名
)
```

您也可以使用命令行工具指定要使用的自定义函数：

```bash
python examples/RewardSQL/train_verl.py \
    --config examples/RewardSQL/verl/grpo_trainer.yaml \
    --algorithm grpo \
    --data_path path/to/data \
    --output_dir outputs/verl_train \
    --core_algos examples.RewardSQL.verl.core_algos \
    --core_function compute_weighted_grpo_advantage
```

框架会根据指定的函数名自动替换VERL中的优势计算函数。如果不指定函数名，系统会自动搜索模块中适合当前算法类型的函数。

## 配置文件

配置文件是YAML格式，包含以下主要部分：

1. **data**：数据配置，包括训练文件、批大小等
2. **actor_rollout_ref**：演员-rollout-参考策略配置
3. **critic**：评论家模型配置（PPO需要）
4. **reward_model**：奖励模型配置
5. **verifier**：验证器配置
6. **algorithm**：算法配置
7. **trainer**：训练过程配置

具体配置示例请参考`examples/RewardSQL/verl/grpo_trainer.yaml`。

## 示例

完整示例请参考`examples/RewardSQL/train_verl.py`。

## 扩展方法

### 添加新算法

1. 在`verl/recipe/`下创建新算法目录
2. 实现核心算法模块（优势函数等）
3. 实现训练器类
4. 在`VERLTrainingManager`中添加对新算法的支持

### 添加新组件类型

1. 在`component_registry.py`中添加新组件类型
2. 添加对应的注册和获取方法
3. 更新`__init__.py`以导出新方法