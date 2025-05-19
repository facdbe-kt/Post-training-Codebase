RewardSQL: zyx
SQL_optimizer: wpy
Askdata: zyx

# 示例应用

本目录包含使用大型语言模型训练与推理框架的实际应用示例，展示了如何利用框架的各个组件解决具体任务。

## 目录结构

```
examples/
├── RewardSQL/            # SQL指令微调与强化学习训练
│   ├── sft_config.json   # SFT配置文件
│   ├── rm_config.json    # 奖励模型配置文件
│   ├── train_verl.py     # VERL强化学习训练脚本
│   └── verl/             # VERL自定义组件
├── SQL_optimizer/        # SQL优化工具应用
└── DeepData/             # 数据分析与可视化
```

## 示例介绍

### RewardSQL

RewardSQL是一个完整的SQL指令任务训练流程示例，演示了如何从监督微调、奖励模型训练到强化学习训练的全流程实现。这个示例特别展示了：

1. **核心算法自定义**：展示了如何创建和使用自定义优势计算函数（如`compute_grpo_process_advantage_avg_unique`等）
2. **组件注册系统**：通过装饰器注册自定义组件（奖励函数、奖励管理器等）
3. **训练管理器接口**：使用`VERLTrainingManager`简化VERL训练流程

特别适合需要开发SQL生成与优化任务的用户。详细文档请查看[RewardSQL/README.md](./RewardSQL/README.md)。

**使用示例**：
```bash
# 训练流程
# 1. 首先进行监督微调
python src/initialization/supervised_finetune/sft.py --config examples/RewardSQL/sft_config.json

# 2. 接着训练奖励模型
python src/initialization/reward_model_training/rm_training.py --config examples/RewardSQL/rm_config.json

# 3. 最后进行强化学习训练
python examples/RewardSQL/train_verl.py \
    --config examples/RewardSQL/verl/grpo_trainer.yaml \
    --algorithm grpo \
    --data_path path/to/data \
    --output_dir outputs/verl_train \
    --core_function compute_weighted_grpo_advantage
```

### SQL_optimizer

SQL_optimizer是一个专注于SQL查询优化的应用示例，主要演示：

1. **推理框架应用**：使用框架的推理组件进行SQL优化
2. **解码策略比较**：展示不同解码策略（如贪婪搜索、集束搜索、Best-of-N等）对SQL优化的影响
3. **推理加速技术**：演示BatchDecode和分布式推理等技术

特别适合构建SQL优化服务的用户。

**使用示例**：
```python
from examples.SQL_optimizer import SQLOptimizer

# 创建优化器实例
optimizer = SQLOptimizer(model_path="path/to/model")

# 优化SQL查询
optimized_sql = optimizer.optimize(
    query="SELECT * FROM users WHERE name = 'John'",
    decoding_strategy="beam_search",
    num_beams=5
)
```

### DeepData

DeepData是一个数据分析与可视化应用示例，展示了如何使用模型处理和分析复杂数据。主要特点：

1. **数据解析与提取**：使用模型从非结构化文本中提取结构化信息
2. **可视化生成**：基于数据分析结果生成可视化代码
3. **分析推理链**：通过思维链(Chain-of-Thought)等技术提高分析质量

适合需要进行数据分析与可视化的场景。

**使用示例**：
```python
from examples.DeepData import DataAnalyzer

# 创建分析器实例
analyzer = DataAnalyzer(model_path="path/to/model")

# 分析数据
result = analyzer.analyze(
    data="path/to/data.csv",
    question="What's the trend of sales in the last quarter?"
)

# 获取可视化代码
viz_code = analyzer.generate_visualization(result)
```

## 如何使用这些示例

这些示例提供了完整的工作流程和可复用代码，您可以：

1. **直接运行**：按照示例中的指令直接运行，体验完整流程
2. **修改配置**：调整配置文件适应您自己的数据和任务
3. **借鉴架构**：参考示例中的代码架构，构建自己的应用
4. **扩展功能**：基于示例添加新功能，如新的奖励函数或解码策略

## 开发自己的示例

如果您想开发自己的应用示例，建议：

1. 创建一个新的子目录
2. 提供完整的README.md文档
3. 尽量模块化设计，将核心功能封装为可复用组件
4. 编写演示脚本，展示主要功能
5. 提供使用说明和示例配置文件

更多框架使用指南，请参考项目根目录的[README.md](../README.md)。