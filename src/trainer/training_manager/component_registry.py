"""
组件注册器

提供组件的注册和获取功能，以便在训练过程中动态使用自定义组件
"""

import inspect
from typing import Dict, Any, Type, Optional, Callable, List, Union

class ComponentRegistry:
    """
    组件注册器
    
    管理训练过程中可能用到的各种自定义组件，包括奖励函数、算法、模型等
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化注册表"""
        # 组件类型 -> 名称 -> 实现
        self.registry = {
            "reward_manager": {},
            "reward_function": {},
            "core_algos": {},
            "custom_model": {},
            "algorithm": {},
        }
    
    def register_reward_manager(self, name: str = None):
        """
        注册奖励管理器类
        
        Args:
            name: 组件名称，如果为None，则使用类名
        """
        def decorator(cls):
            component_name = name or cls.__name__
            self.registry["reward_manager"][component_name] = cls
            return cls
        return decorator
    
    def register_reward_function(self, name: str = None):
        """
        注册奖励函数
        
        Args:
            name: 组件名称，如果为None，则使用函数名
        """
        def decorator(func):
            component_name = name or func.__name__
            self.registry["reward_function"][component_name] = func
            return func
        return decorator
    
    def register_core_algos(self, name: str = None):
        """
        注册核心算法模块
        
        Args:
            name: 组件名称，如果为None，则使用模块名
        """
        def decorator(module):
            component_name = name or module.__name__
            self.registry["core_algos"][component_name] = module
            return module
        return decorator
    
    def register_custom_model(self, name: str = None):
        """
        注册自定义模型
        
        Args:
            name: 组件名称，如果为None，则使用类名
        """
        def decorator(cls):
            component_name = name or cls.__name__
            self.registry["custom_model"][component_name] = cls
            return cls
        return decorator
    
    def register_algorithm(self, name: str = None):
        """
        注册训练算法
        
        Args:
            name: 组件名称，如果为None，则使用类名
        """
        def decorator(cls):
            component_name = name or cls.__name__
            self.registry["algorithm"][component_name] = cls
            return cls
        return decorator
    
    def register(self, component_type: str, name: str, component: Any):
        """
        手动注册组件
        
        Args:
            component_type: 组件类型
            name: 组件名称
            component: 组件实现
        """
        if component_type not in self.registry:
            raise ValueError(f"未知的组件类型: {component_type}")
        self.registry[component_type][name] = component
    
    def get_reward_manager(self, name: str) -> Optional[Type]:
        """
        获取奖励管理器类
        
        Args:
            name: 组件名称
            
        Returns:
            奖励管理器类，如果不存在则返回None
        """
        return self.registry["reward_manager"].get(name)
    
    def get_reward_function(self, name: str) -> Optional[Callable]:
        """
        获取奖励函数
        
        Args:
            name: 组件名称
            
        Returns:
            奖励函数，如果不存在则返回None
        """
        return self.registry["reward_function"].get(name)
    
    def get_core_algos(self, name: str) -> Optional[Any]:
        """
        获取核心算法模块
        
        Args:
            name: 组件名称
            
        Returns:
            核心算法模块，如果不存在则返回None
        """
        return self.registry["core_algos"].get(name)
    
    def get_custom_model(self, name: str) -> Optional[Type]:
        """
        获取自定义模型类
        
        Args:
            name: 组件名称
            
        Returns:
            自定义模型类，如果不存在则返回None
        """
        return self.registry["custom_model"].get(name)
    
    def get_algorithm(self, name: str) -> Optional[Type]:
        """
        获取训练算法类
        
        Args:
            name: 组件名称
            
        Returns:
            训练算法类，如果不存在则返回None
        """
        return self.registry["algorithm"].get(name)
    
    def get(self, component_type: str, name: str) -> Optional[Any]:
        """
        获取指定类型和名称的组件
        
        Args:
            component_type: 组件类型
            name: 组件名称
            
        Returns:
            组件，如果不存在则返回None
        """
        if component_type not in self.registry:
            raise ValueError(f"未知的组件类型: {component_type}")
        return self.registry[component_type].get(name)
    
    def get_component_function(self, component_type: str, function_name: str) -> Optional[Callable]:
        """
        获取指定类型组件中的特定函数
        
        Args:
            component_type: 组件类型
            function_name: 函数名称
            
        Returns:
            函数对象，如果不存在则返回None
        """
        if component_type not in self.registry:
            raise ValueError(f"未知的组件类型: {component_type}")
        
        for component_name, component in self.registry[component_type].items():
            if hasattr(component, function_name):
                return getattr(component, function_name)
            # 如果组件是模块，检查模块内的函数
            elif inspect.ismodule(component):
                if hasattr(component, function_name):
                    return getattr(component, function_name)
        
        return None
    
    def list_components(self, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        列出已注册的组件
        
        Args:
            component_type: 组件类型，如果为None则列出所有类型
            
        Returns:
            组件类型 -> 组件名称列表的字典
        """
        if component_type is not None:
            if component_type not in self.registry:
                raise ValueError(f"未知的组件类型: {component_type}")
            return {component_type: list(self.registry[component_type].keys())}
        
        return {k: list(v.keys()) for k, v in self.registry.items()}


# 创建全局注册器实例
registry = ComponentRegistry()

# 导出便捷的装饰器函数
register_reward_manager = registry.register_reward_manager
register_reward_function = registry.register_reward_function
register_core_algos = registry.register_core_algos
register_custom_model = registry.register_custom_model
register_algorithm = registry.register_algorithm 