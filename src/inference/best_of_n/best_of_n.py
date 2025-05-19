import os
import sys
from typing import Dict, Any, List, Union, Optional, Tuple
import abc

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import InferenceBase
from models import ModelLoader

class BestOfN(InferenceBase):
    """
    Best-of-N解码策略类
    生成多个候选输出，然后根据评分函数选择最佳结果
    """
    
    def __init__(self, model_name_or_path: str, sampling_method: str = "best_of_n", hyperparameters: Dict[str, Any] = None):
        """
        初始化Best-of-N解码策略
        
        Args:
            model_name_or_path: 模型名称或路径，支持HuggingFace模型ID、本地路径或URL
            sampling_method: 采样方法名称，默认为"best_of_n"
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
            "temperature": 0.7,  # 使用较高的温度以增加多样性
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None,
            "n": 5,  # 生成候选数量
        }
        
        # 将default_params中的参数合并到hyperparameters中，优先使用hyperparameters中的值
        for k, v in self.default_params.items():
            if k not in self.hyperparameters:
                self.hyperparameters[k] = v
    
    def score(self, candidate: str, context: Dict[str, Any] = None) -> float:
        """
        对候选输出进行评分的虚函数，需要在子类中实现
        
        Args:
            candidate: 候选输出
            context: 评分上下文信息
            
        Returns:
            float: 候选输出的得分，分数越高表示质量越好
        """
        # 默认实现：返回长度作为得分（越长越好）
        # 这只是一个简单的示例，实际应用中应该根据任务需求实现更复杂的评分函数
        return len(candidate)
    
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
        
        # 根据模型类型生成多个候选输出
        if self.model_type == "huggingface":
            candidates = self._forward_huggingface(input_text)
        elif self.model_type == "vllm":
            candidates = self._forward_vllm(input_text)
        elif self.model_type == "openai":
            candidates = self._forward_openai(input_text)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 对候选输出进行评分
        scored_candidates = [(candidate, self.score(candidate, {"input": input_text})) for candidate in candidates]
        
        # 按照分数从高到低排序
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        
        # 返回得分最高的候选输出
        best_candidate = sorted_candidates[0][0]
        
        return best_candidate, {
            "candidates": candidates,
            "scores": [score for _, score in sorted_candidates],
            "best_candidate_index": candidates.index(best_candidate)
        }
    
    def _forward_huggingface(self, input_text: str) -> List[str]:
        """使用HuggingFace模型生成多个候选输出"""
        import torch
        
        model, tokenizer = self.model
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成多个候选输出
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.hyperparameters["max_tokens"],
                do_sample=True,
                temperature=self.hyperparameters["temperature"],
                top_p=self.hyperparameters["top_p"],
                num_return_sequences=self.hyperparameters["n"],
                num_beams=1,  # 不使用beam search
            )
        
        # 解码输出
        input_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        candidates = []
        
        for seq in output:
            generated_text = tokenizer.decode(seq, skip_special_tokens=True)
            # 移除输入文本部分
            result_text = generated_text[input_length:]
            candidates.append(result_text)
        
        return candidates
    
    def _forward_vllm(self, input_text: str) -> List[str]:
        """使用VLLM模型生成多个候选输出"""
        try:
            from vllm import SamplingParams
            
            # 设置采样参数
            sampling_params = SamplingParams(
                max_tokens=self.hyperparameters["max_tokens"],
                temperature=self.hyperparameters["temperature"],
                top_p=self.hyperparameters["top_p"],
                n=self.hyperparameters["n"],
                stop=self.hyperparameters.get("stop", None)
            )
            
            # 执行推理
            outputs = self.model.generate(input_text, sampling_params)
            candidates = [output.text for output in outputs[0].outputs]
            
            return candidates
        except ImportError:
            raise ImportError("VLLM 未安装，请通过 'pip install vllm' 安装")
    
    def _forward_openai(self, input_text: str) -> List[str]:
        """使用OpenAI API生成多个候选输出"""
        try:
            # 构造请求参数
            params = {
                "model": self.hyperparameters.get("model", "gpt-3.5-turbo"),
                "prompt": input_text,
                "max_tokens": self.hyperparameters["max_tokens"],
                "temperature": self.hyperparameters["temperature"],
                "top_p": self.hyperparameters["top_p"],
                "n": self.hyperparameters["n"],
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
            candidates = [choice.text for choice in response.choices]
            
            return candidates
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