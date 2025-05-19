# Inference 推理框架

一个灵活、可扩展的通用推理框架，支持多种模型源和解码策略。本框架设计用于简化大语言模型的推理过程，提供统一的接口和丰富的功能。

## 主要特性

- **通用模型支持**：支持HuggingFace模型、VLLM、OpenAI API等多种模型源
- **自动模型检测**：可以自动检测模型类型（本地路径、URL、HuggingFace ID等）
- **可扩展的解码策略**：内置多种解码策略，包括贪婪解码、Best-of-N选择和自我改进
- **任务特定优化**：支持扩展特定任务（如SQL生成）的优化策略
- **批量处理**：支持批量推理，提高处理效率
- **统一接口**：所有解码策略共享相同的API接口，便于集成和扩展

## 架构设计

```
inference/
├── __init__.py          # 包初始化文件
├── base.py              # 基础推理接口定义
├── models.py            # 模型加载和工具类
├── greedy/              # 贪婪解码策略
│   ├── __init__.py
│   └── greedy.py
├── best_of_n/           # Best-of-N解码策略
│   ├── __init__.py
│   └── best_of_n.py
└── self_improve/        # 自我改进解码策略
    ├── __init__.py
    └── self_improve.py
```

## 快速开始

### 安装依赖

```bash
pip install torch transformers vllm openai tqdm
```

### 基本用法

```python
from src.inference import Greedy, BestOfN, SelfImprove

# 使用贪婪解码策略
greedy_decoder = Greedy(
    model_name_or_path="gpt2",  # 可以是HuggingFace模型ID、本地路径或URL
    hyperparameters={
        "max_tokens": 100,
        "temperature": 0.0
    }
)
result, metadata = greedy_decoder.forward("请为我写一首诗")

# 使用Best-of-N解码策略
best_of_n_decoder = BestOfN(
    model_name_or_path="http://localhost:8000/v1",  # OpenAI API格式的服务
    hyperparameters={
        "api_key": "EMPTY",
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "n": 5  # 生成5个候选
    }
)
result, metadata = best_of_n_decoder.forward("请为我写一首诗")

# 使用自我改进解码策略
self_improve_decoder = SelfImprove(
    model_name_or_path="path/to/local/model",
    hyperparameters={
        "max_tokens": 200,
        "max_iterations": 3  # 最多进行3轮改进
    }
)
result, metadata = self_improve_decoder.forward("请为我写一首诗")
```

## 自定义任务特定解码策略

在实际应用中，可以根据具体任务特点扩展基础解码策略。任务特定的实现应该放在examples目录下，而不是src中。例如SQL任务的示例：

```python
# examples/RewardSQL/sql_decoders.py
from src.inference import BestOfN, SelfImprove

class SQLBestOfN(BestOfN):
    """SQL任务特定的Best-of-N解码策略类"""
    
    def score(self, candidate: str, context: Dict[str, Any] = None) -> float:
        """为SQL候选输出打分的自定义实现"""
        # SQL特定的评分逻辑
        score = 0
        
        # 检查SQL关键字
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN"]
        keyword_count = sum(1 for keyword in sql_keywords if keyword in candidate.upper())
        score += keyword_count * 2
        
        return score

# 使用方法
sql_best_of_n = SQLBestOfN(
    model_name_or_path="path/to/sql/model",
    hyperparameters={
        "max_tokens": 256,
        "temperature": 0.7,
        "n": 10  # 生成10个SQL候选
    }
)
```

## 自定义解码策略

您可以通过继承`InferenceBase`基类来创建自定义的解码策略：

```python
from src.inference import InferenceBase
from typing import Dict, Any, List, Tuple, Optional

class MyCustomDecoder(InferenceBase):
    def __init__(self, model_name_or_path: str, sampling_method: str = "custom", hyperparameters: Dict[str, Any] = None):
        # 初始化逻辑
        pass
    
    def forward(self, input_text: str, prompt_template: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        # 自定义推理逻辑
        pass
    
    def batch_forward(self, input_texts: List[str], prompt_template: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        # 自定义批量推理逻辑
        pass
```

## 高级功能

### 模板应用

所有解码策略都支持提示模板：

```python
template = "请用中文回答以下问题：\n{input}"
result, metadata = decoder.forward("What is the capital of France?", prompt_template=template)
```

### 批量处理

批量处理多个输入：

```python
inputs = ["问题1", "问题2", "问题3"]
results = decoder.batch_forward(inputs)
```

### 超参数调优

每种解码策略都可以通过`hyperparameters`参数进行详细配置：

```python
hyperparameters = {
    # 通用参数
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": ["\n\n"],
    
    # Best-of-N特定参数
    "n": 5,
    
    # 自我改进特定参数
    "max_iterations": 3,
    "improvement_threshold": 0.05,
    "critique_template": "自定义评估模板...",
    "improve_template": "自定义改进模板...",
    
    # OpenAI API特定参数
    "api_key": "your-api-key",
    "api_base": "custom-api-endpoint",
    "model": "specific-model-name"
}
```

## 性能优化

- 对于大规模推理任务，建议使用VLLM作为后端以获得更高的吞吐量
- 对于批处理场景，可以使用`batch_forward`方法代替循环调用`forward`
- 对于特定任务，选择适合的特定解码策略可以获得更好的结果

## 扩展开发

本框架设计为易于扩展，您可以：

1. 添加新的模型后端支持
2. 实现新的解码策略
3. 为特定任务开发优化的评分函数（放在examples下）
4. 扩展现有功能，如添加缓存、流式输出等

## 许可证

MIT License 