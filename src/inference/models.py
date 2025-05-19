import os
import re
import json
from typing import Dict, Any, List, Union, Optional, Tuple
import importlib.util
from urllib.parse import urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

class ModelLoader:
    """
    模型加载器类，负责加载和管理不同来源的模型
    支持从本地路径、HuggingFace Hub或自定义URL加载模型
    """
    
    @staticmethod
    def is_url(path: str) -> bool:
        """
        检查给定的路径是否为URL
        
        Args:
            path: 要检查的路径
            
        Returns:
            bool: 如果是URL则返回True，否则返回False
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def is_local_path(path: str) -> bool:
        """
        检查给定的路径是否为本地路径
        
        Args:
            path: 要检查的路径
            
        Returns:
            bool: 如果是本地路径则返回True，否则返回False
        """
        return os.path.exists(path)
    
    @staticmethod
    def detect_model_type(model_name_or_path: str) -> str:
        """
        自动检测模型类型
        
        Args:
            model_name_or_path: 模型名称或路径
            
        Returns:
            str: 模型类型，例如'huggingface', 'openai', 'vllm'等
        """
        # 根据模型路径或名称特征来检测模型类型
        if ModelLoader.is_url(model_name_or_path):
            if "api.openai.com" in model_name_or_path:
                return "openai"
            # 可以添加其他API的检测逻辑
            return "api"
        
        if ModelLoader.is_local_path(model_name_or_path):
            # 检查本地模型类型
            if os.path.exists(os.path.join(model_name_or_path, "config.json")):
                return "huggingface"
            # 检查是否为VLLM支持的格式
            return "vllm"
        
        # 默认视为HuggingFace模型ID
        return "huggingface"
    
    @staticmethod
    def load_huggingface_model(model_name_or_path: str, device: str = None, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        加载HuggingFace模型
        
        Args:
            model_name_or_path: 模型名称或路径
            device: 设备，例如'cuda', 'cpu'等
            kwargs: 其他参数
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: 加载的模型和分词器
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            **kwargs
        )
        
        return model, tokenizer
    
    @staticmethod
    def load_vllm_model(model_name_or_path: str, **kwargs) -> Any:
        """
        加载VLLM模型
        
        Args:
            model_name_or_path: 模型名称或路径
            kwargs: 其他参数
            
        Returns:
            Any: 加载的VLLM模型实例
        """
        try:
            from vllm import LLM
            return LLM(
                model=model_name_or_path,
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
                dtype=kwargs.get("dtype", "float16"),
                **{k: v for k, v in kwargs.items() if k not in ["gpu_memory_utilization", "dtype"]}
            )
        except ImportError:
            raise ImportError("VLLM 未安装，请通过 'pip install vllm' 安装")
    
    @staticmethod
    def load_openai_client(api_key: str = None, api_base: str = None, **kwargs) -> Any:
        """
        加载OpenAI API客户端
        
        Args:
            api_key: OpenAI API密钥
            api_base: OpenAI API基础URL
            kwargs: 其他参数
            
        Returns:
            Any: OpenAI客户端
        """
        try:
            from openai import OpenAI
            
            # 优先使用传入的参数，否则尝试从环境变量获取
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
            
            if api_base is None:
                api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            return OpenAI(
                api_key=api_key,
                base_url=api_base,
                **kwargs
            )
        except ImportError:
            raise ImportError("OpenAI 未安装，请通过 'pip install openai' 安装")
    
    @staticmethod
    def load_model(model_name_or_path: str, model_type: str = None, **kwargs) -> Tuple[Any, str]:
        """
        通用模型加载函数，支持多种模型源
        
        Args:
            model_name_or_path: 模型名称、本地路径或URL
            model_type: 模型类型，如果为None则自动检测
            kwargs: 其他参数
            
        Returns:
            Tuple[Any, str]: (加载的模型对象, 模型类型)
        """
        # 如果未指定模型类型，则自动检测
        if model_type is None:
            model_type = ModelLoader.detect_model_type(model_name_or_path)
        
        # 根据模型类型加载相应的模型
        if model_type == "huggingface":
            model = ModelLoader.load_huggingface_model(model_name_or_path, **kwargs)
        elif model_type == "vllm":
            model = ModelLoader.load_vllm_model(model_name_or_path, **kwargs)
        elif model_type == "openai":
            # 对于OpenAI API，直接返回客户端
            model = ModelLoader.load_openai_client(
                api_key=kwargs.get("api_key"),
                api_base=kwargs.get("api_base"),
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "api_base"]}
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model, model_type


class Utils:
    """
    工具类，提供各种辅助函数
    """
    
    @staticmethod
    def extract_sql_from_str(input_data: str) -> List[str]:
        """
        从字符串中提取SQL代码块
        
        Args:
            input_data: 输入字符串
            
        Returns:
            List[str]: 提取的SQL代码列表
        """
        sql_code_blocks = re.findall(r'```sql\s(.*?)```', input_data, re.DOTALL)
        # 扁平化列表并去除多余空格
        extracted_sql = [code.strip() for code in sql_code_blocks]
        return extracted_sql
    
    @staticmethod
    def save_results(results: List[Dict[str, Any]], output_path: str, filename: str = "results.json") -> None:
        """
        保存结果到JSON文件
        
        Args:
            results: 结果列表
            output_path: 输出路径
            filename: 文件名
        """
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, filename), "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False) 