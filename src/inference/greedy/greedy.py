import os
import sys
from typing import Dict, Any, List, Union, Optional, Tuple

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import InferenceBase
from models import ModelLoader

class Greedy(InferenceBase):
    """
    贪婪解码策略类
    使用贪婪解码方式进行推理，即在每个时间步选择概率最高的token
    """
    
    def __init__(self, model_name_or_path: str, sampling_method: str = "greedy", hyperparameters: Dict[str, Any] = None):
        """
        初始化贪婪解码策略
        
        Args:
            model_name_or_path: 模型名称或路径，支持HuggingFace模型ID、本地路径或URL
            sampling_method: 采样方法名称，默认为"greedy"
            hyperparameters: 超参数字典
        """
        self.model_name_or_path = model_name_or_path
        self.sampling_method = sampling_method
        self.hyperparameters = hyperparameters or {}
        
        # 加载模型
        self.model, self.model_type = ModelLoader.load_model(
            model_name_or_path, 
            **self.hyperparameters
        )
        
        # 根据模型类型设置默认参数
        self._set_default_params()
    
    def _set_default_params(self):
        """设置默认参数"""
        self.default_params = {
            "max_tokens": 256,
            "temperature": 0.0,  # 贪婪搜索通常使用温度为0
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None
        }
        
        # 将default_params中的参数合并到hyperparameters中，优先使用hyperparameters中的值
        for k, v in self.default_params.items():
            if k not in self.hyperparameters:
                self.hyperparameters[k] = v
    
    def forward(self, input_text: str, prompt_template: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        执行推理
        
        Args:
            input_text: 输入文本
            prompt_template: 可选的提示模板
            
        Returns:
            Tuple[str, Dict[str, Any]]: 返回推理结果和附加信息
        """
        # 如果有提示模板，则应用模板
        if prompt_template:
            input_text = prompt_template.format(input=input_text)
        
        if self.model_type == "huggingface":
            return self._forward_huggingface(input_text)
        elif self.model_type == "vllm":
            return self._forward_vllm(input_text)
        elif self.model_type == "openai":
            return self._forward_openai(input_text)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _forward_huggingface(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """使用HuggingFace模型进行推理"""
        import torch
        
        model, tokenizer = self.model
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.hyperparameters["max_tokens"],
                do_sample=False,  # 贪婪搜索，不进行采样
                num_return_sequences=1
            )
        
        # 解码输出
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 移除输入文本部分
        result_text = generated_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return result_text, {"raw_output": generated_text}
    
    def _forward_vllm(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """使用VLLM模型进行推理"""
        try:
            from vllm import SamplingParams
            
            # 设置采样参数
            sampling_params = SamplingParams(
                max_tokens=self.hyperparameters["max_tokens"],
                temperature=0.0,  # 贪婪搜索使用温度为0
                top_p=1.0,
                stop=self.hyperparameters.get("stop", None)
            )
            
            # 执行推理
            outputs = self.model.generate(input_text, sampling_params)
            result = outputs[0].outputs[0].text
            
            return result, {"raw_output": result}
        except ImportError:
            raise ImportError("VLLM 未安装，请通过 'pip install vllm' 安装")
    
    def _forward_openai(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """使用OpenAI API进行推理"""
        try:
            # 构造请求参数
            params = {
                "model": self.hyperparameters.get("model", "gpt-3.5-turbo"),
                "prompt": input_text,
                "max_tokens": self.hyperparameters["max_tokens"],
                "temperature": 0.0,  # 贪婪搜索使用温度为0
                "top_p": 1.0,
                "n": 1,
                "stream": False
            }
            
            # 添加可选参数
            if self.hyperparameters.get("stop"):
                params["stop"] = self.hyperparameters["stop"]
            
            if self.hyperparameters.get("frequency_penalty"):
                params["frequency_penalty"] = self.hyperparameters["frequency_penalty"]
                
            if self.hyperparameters.get("presence_penalty"):
                params["presence_penalty"] = self.hyperparameters["presence_penalty"]
            
            # 执行API调用
            response = self.model.completions.create(**params)
            result = response.choices[0].text
            
            return result, {"raw_response": response}
        except Exception as e:
            raise RuntimeError(f"OpenAI API调用失败: {str(e)}")
    
    def batch_forward(self, input_texts: List[str], prompt_template: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        批量执行推理
        
        Args:
            input_texts: 输入文本列表
            prompt_template: 可选的提示模板
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: 返回推理结果和附加信息的列表
        """
        results = []
        for input_text in input_texts:
            result = self.forward(input_text, prompt_template)
            results.append(result)
        return results 