import abc
from typing import Dict, Any, Tuple, List, Optional

class InferenceBase(abc.ABC):
    """
    基础推理抽象类
    所有的推理策略都应该继承这个类并实现相应的方法
    """
    
    @abc.abstractmethod
    def __init__(self, model_name_or_path: str, sampling_method: str, hyperparameters: Dict[str, Any] = None):
        """
        初始化推理基类
        
        Args:
            model_name_or_path: 模型名称或路径，支持HuggingFace模型ID、本地路径或URL
            sampling_method: 采样方法名称
            hyperparameters: 超参数字典
        """
        pass
    
    @abc.abstractmethod
    def forward(self, input_text: str, prompt_template: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        执行推理
        
        Args:
            input_text: 输入文本
            prompt_template: 可选的提示模板
            
        Returns:
            Tuple[str, Dict[str, Any]]: 返回推理结果和附加信息
        """
        pass