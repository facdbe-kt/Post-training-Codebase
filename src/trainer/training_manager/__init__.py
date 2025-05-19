"""
训练管理器模块

提供通用的训练管理接口，支持不同的训练算法和模型
"""

from .base import BaseTrainingManager
from .rl_training_manager import RLTrainingManager
from .verl_training_manager import VERLTrainingManager
from .component_registry import (
    registry, 
    register_reward_manager,
    register_reward_function, 
    register_core_algos,
    register_custom_model, 
    register_algorithm
)

__all__ = [
    "BaseTrainingManager",
    "RLTrainingManager",
    "VERLTrainingManager",
    "registry",
    "register_reward_manager",
    "register_reward_function",
    "register_core_algos",
    "register_custom_model",
    "register_algorithm",
    "get_component_function"
]

# 导出便捷函数
def get_component_function(component_type, function_name):
    """获取指定类型组件中的特定函数"""
    return registry.get_component_function(component_type, function_name)
