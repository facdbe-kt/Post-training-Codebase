"""
基础训练管理器

定义通用的训练接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import os
import logging

logger = logging.getLogger(__name__)

class BaseTrainingManager(ABC):
    """
    基础训练管理器类
    
    提供通用的训练接口和基础功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练管理器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        初始化训练资源、模型等
        """
        pass
    
    @abstractmethod
    def train(self, 
              data_path: Union[str, List[str]], 
              output_dir: str, 
              **kwargs) -> Dict[str, Any]:
        """
        执行训练过程
        
        Args:
            data_path: 训练数据路径
            output_dir: 输出目录
            **kwargs: 额外参数
            
        Returns:
            训练结果统计信息
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                 data_path: Union[str, List[str]], 
                 **kwargs) -> Dict[str, Any]:
        """
        执行评估过程
        
        Args:
            data_path: 评估数据路径
            **kwargs: 额外参数
            
        Returns:
            评估结果统计信息
        """
        pass
    
    @abstractmethod
    def save(self, 
             output_dir: str, 
             step: Optional[int] = None) -> str:
        """
        保存训练状态和模型
        
        Args:
            output_dir: 输出目录
            step: 当前步骤编号
            
        Returns:
            保存的路径
        """
        pass
        
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        加载训练状态和模型
        
        Args:
            model_path: 模型路径
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        Returns:
            当前配置的副本
        """
        return self.config.copy()
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_updates: 要更新的配置项
        """
        self.config.update(config_updates) 