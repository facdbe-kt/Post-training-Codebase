from typing import Dict, Any, Tuple, Optional

from ..base import InferenceBase
from ..models import ModelLoader

class DirectInference(InferenceBase):
    def __init__(self, model_name_or_path: str, hyperparameters: Dict[str, Any]):
        self.model_name_or_path = model_name_or_path
        self.key_file = hyperparameters['key_file']
        self.temp = hyperparameters['temperature']
        self.max_tokens = hyperparameters['max_tokens']
        self.base_url = hyperparameters['api_base']

        self.hyperparameters = hyperparameters
        self.model, self.model_type = ModelLoader.load_model(
            model_name_or_path, 
            model_type=None,
            hyperparameters=hyperparameters
        )
    
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
                do_sample=self.hyperparameters["temperature"] > 0,
                num_return_sequences=1,
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
                temperature=self.hyperparameters["temperature"],
                top_p=self.hyperparameters.get("top_p", 1.0),
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
            params = {
                "model": self.hyperparameters.get("model", "gpt-3.5-turbo"),
                "prompt": input_text,
                "max_tokens": self.hyperparameters["max_tokens"],
                "temperature": self.hyperparameters["temperature"],
                "top_p": self.hyperparameters.get("top_p", 1.0),
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