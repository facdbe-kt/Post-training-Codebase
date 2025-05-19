"""
监督微调训练脚本
支持通过命令行参数或配置文件设置超参数
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import torch
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments

logger = get_logger()


@dataclass
class SFTConfig:
    """监督微调配置类"""
    # 模型配置
    model_id_or_path: str = "Qwen/Qwen2.5-3B-Instruct"  # 模型ID或路径
    system: str = "You are a helpful assistant."  # 系统提示词
    output_dir: str = "output"  # 输出目录

    # 数据集配置
    dataset: List[str] = field(default_factory=lambda: ["AI-ModelScope/alpaca-gpt4-data-zh#500", 
                                                       "AI-ModelScope/alpaca-gpt4-data-en#500"])
    data_seed: int = 42  # 数据随机种子
    max_length: int = 2048  # 最大序列长度
    split_dataset_ratio: float = 0.01  # 验证集拆分比例
    num_proc: int = 4  # 数据处理的进程数
    model_name: List[str] = field(default_factory=lambda: ["小黄", "Xiao Huang"])  # 模型名称（中英文）
    model_author: List[str] = field(default_factory=lambda: ["魔搭", "ModelScope"])  # 模型作者（中英文）

    # LoRA配置
    lora_rank: int = 8  # LoRA秩
    lora_alpha: int = 32  # LoRA alpha

    # 训练配置
    learning_rate: float = 1e-4  # 学习率
    per_device_train_batch_size: int = 1  # 每个设备的训练批量大小
    per_device_eval_batch_size: int = 1  # 每个设备的评估批量大小
    gradient_checkpointing: bool = True  # 是否启用梯度检查点
    weight_decay: float = 0.1  # 权重衰减
    lr_scheduler_type: str = "cosine"  # 学习率调度器类型
    warmup_ratio: float = 0.05  # 预热比例
    save_strategy: str = "steps"  # 保存策略
    save_steps: int = 50  # 保存步数
    eval_strategy: str = "steps"  # 评估策略
    eval_steps: int = 50  # 评估步数
    gradient_accumulation_steps: int = 16  # 梯度累积步数
    num_train_epochs: int = 1  # 训练轮数
    metric_for_best_model: str = "loss"  # 最佳模型的指标
    save_total_limit: int = 2  # 保存的检查点总数
    logging_steps: int = 5  # 日志记录步数
    dataloader_num_workers: int = 1  # 数据加载器的工作线程数
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])  # 报告工具
    torch_dtype: str = "bfloat16"  # 模型精度
    deepspeed_config: Optional[str] = None  # DeepSpeed配置文件路径
    device: str = "cuda"  # 设备
    seed: int = 42  # 随机种子

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SFTConfig":
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "SFTConfig":
        """从JSON文件读取配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: str) -> None:
        """保存配置到JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def train_sft(config: SFTConfig) -> str:
    """
    执行监督微调训练
    
    Args:
        config: 训练配置
        
    Returns:
        str: 最后一个模型检查点的路径
    """
    # 设置随机种子
    seed_everything(config.seed)
    
    # 确保输出目录是绝对路径
    output_dir = os.path.abspath(os.path.expanduser(config.output_dir))
    logger.info(f'输出目录: {output_dir}')
    
    # 获取模型和分词器
    model, tokenizer = get_model_tokenizer(config.model_id_or_path)
    logger.info(f'模型信息: {model.model_info}')
    
    # 获取模板
    template = get_template(model.model_meta.template, 
                           tokenizer, 
                           default_system=config.system, 
                           max_length=config.max_length)
    template.set_mode('train')
    
    # 设置LoRA配置
    target_modules = find_all_linears(model)
    lora_config = LoraConfig(
        task_type='CAUSAL_LM', 
        r=config.lora_rank, 
        lora_alpha=config.lora_alpha,
        target_modules=target_modules
    )
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'LoRA配置: {lora_config}')
    
    # 打印模型结构和可训练参数
    logger.info(f'模型: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'模型参数信息: {model_parameter_info}')
    
    # 加载和准备数据集
    train_dataset, val_dataset = load_dataset(
        config.dataset, 
        split_dataset_ratio=config.split_dataset_ratio, 
        num_proc=config.num_proc,
        model_name=config.model_name, 
        model_author=config.model_author, 
        seed=config.data_seed
    )
    
    logger.info(f'训练集大小: {len(train_dataset)}')
    logger.info(f'验证集大小: {len(val_dataset)}')
    logger.info(f'训练集样例: {train_dataset[0]}')
    
    # 准备数据集
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=config.num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=config.num_proc)
    logger.info(f'编码后的训练样例: {train_dataset[0]}')
    
    # 打印样例
    template.print_inputs(train_dataset[0])
    
    # 设置训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_checkpointing=config.gradient_checkpointing,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        report_to=config.report_to,
        logging_first_step=True,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        metric_for_best_model=config.metric_for_best_model,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        data_seed=config.data_seed,
    )
    
    # 如果指定了DeepSpeed配置，则添加到训练参数
    if config.deepspeed_config:
        training_args.deepspeed = config.deepspeed_config
    
    # 设置模型精度
    if config.torch_dtype == "bfloat16":
        training_args.bf16 = True
    elif config.torch_dtype == "float16":
        training_args.fp16 = True
    
    model.enable_input_require_grads()  # 兼容梯度检查点
    
    # 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    
    # 开始训练
    logger.info("开始训练")
    trainer.train()
    
    # 获取最后一个检查点
    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'最后一个模型检查点: {last_model_checkpoint}')
    
    # 保存配置
    config.to_json(os.path.join(output_dir, "sft_config.json"))
    
    return last_model_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="监督微调训练")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model_id_or_path", type=str, help="模型ID或路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--dataset", type=str, nargs="+", help="数据集列表")
    parser.add_argument("--max_length", type=int, help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, help="每个设备的训练批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, help="训练轮数")
    parser.add_argument("--lora_rank", type=int, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--torch_dtype", type=str, choices=["float32", "float16", "bfloat16"], help="模型精度")
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed配置文件路径")
    parser.add_argument("--seed", type=int, help="随机种子")
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 创建基础配置
    if args.config:
        # 从配置文件加载
        config = SFTConfig.from_json(args.config)
        logger.info(f"从配置文件加载: {args.config}")
    else:
        # 创建默认配置
        config = SFTConfig()
    
    # 使用命令行参数覆盖配置
    for arg_name, arg_value in vars(args).items():
        if arg_name != "config" and arg_value is not None:
            setattr(config, arg_name, arg_value)
    
    # 打印配置
    logger.info(f"训练配置: {config.to_dict()}")
    
    # 开始训练
    last_checkpoint = train_sft(config)
    logger.info(f"训练完成！最后的检查点保存在: {last_checkpoint}")


if __name__ == "__main__":
    main()