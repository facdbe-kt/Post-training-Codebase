"""
奖励模型训练脚本
支持通过命令行参数或配置文件设置超参数
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import torch
from swift.utils import get_logger, seed_everything

logger = get_logger()


@dataclass
class RMConfig:
    """奖励模型配置类"""
    # 模型配置
    model_id_or_path: str = "Qwen/Qwen2.5-3B-Instruct"  # 模型ID或路径
    output_dir: str = "rm_output"  # 输出目录

    # 数据集配置
    dataset: str = "OpenRLHF/preference_dataset_mixture2_and_safe_pku"  # 偏好数据集
    apply_chat_template: bool = True  # 是否应用聊天模板
    chosen_key: str = "chosen"  # 选择文本的字段名
    rejected_key: str = "rejected"  # 拒绝文本的字段名
    max_length: int = 8192  # 最大序列长度
    data_seed: int = 42  # 数据随机种子
    num_proc: int = 4  # 数据处理的进程数
    packing_samples: bool = True  # 是否打包样本

    # 训练配置
    train_batch_size: int = 256  # 训练批量大小
    micro_train_batch_size: int = 1  # 微批量大小
    learning_rate: float = 9e-6  # 学习率
    weight_decay: float = 0.0  # 权重衰减
    warm_up_ratio: float = 0.03  # 预热比例
    max_epochs: int = 1  # 最大训练轮数
    gradient_checkpointing: bool = True  # 是否使用梯度检查点
    gradient_accumulation_steps: int = 16  # 梯度累积步数
    use_wandb: bool = False  # 是否使用Wandb
    wandb_project: str = "reward_model_training"  # Wandb项目名
    save_steps: int = -1  # 保存步数，-1表示每个epoch结束时保存
    logging_steps: int = 1  # 日志记录步数
    eval_steps: int = -1  # 评估步数，-1表示每个epoch结束时评估
    torch_dtype: str = "bfloat16"  # 模型精度
    load_checkpoint: bool = True  # 是否加载最新检查点
    flash_attn: bool = True  # 是否使用Flash Attention
    zero_stage: int = 3  # DeepSpeed ZeRO优化阶段
    seed: int = 42  # 随机种子
    deepspeed_config: Optional[str] = None  # 自定义DeepSpeed配置文件路径

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RMConfig":
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "RMConfig":
        """从JSON文件读取配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: str) -> None:
        """保存配置到JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def train_rm(config: RMConfig) -> str:
    """
    执行奖励模型训练
    
    Args:
        config: 训练配置
        
    Returns:
        str: 模型保存路径
    """
    # 设置随机种子
    seed_everything(config.seed)
    
    # 确保输出目录是绝对路径
    output_dir = os.path.abspath(os.path.expanduser(config.output_dir))
    logger.info(f'输出目录: {output_dir}')
    
    # 构建OpenRLHF命令行
    command = [
        "deepspeed", "--module", "openrlhf.cli.train_rm",
        f"--save_path={output_dir}",
        f"--save_steps={config.save_steps}",
        f"--logging_steps={config.logging_steps}",
        f"--eval_steps={config.eval_steps}",
        f"--train_batch_size={config.train_batch_size}",
        f"--micro_train_batch_size={config.micro_train_batch_size}",
        f"--pretrain={config.model_id_or_path}",
        f"--max_epochs={config.max_epochs}",
        f"--max_len={config.max_length}",
        f"--zero_stage={config.zero_stage}",
        f"--learning_rate={config.learning_rate}",
        f"--dataset={config.dataset}",
        f"--chosen_key={config.chosen_key}",
        f"--rejected_key={config.rejected_key}",
    ]
    
    # 添加可选标志
    if config.apply_chat_template:
        command.append("--apply_chat_template")
    
    if config.torch_dtype == "bfloat16":
        command.append("--bf16")
    elif config.torch_dtype == "float16":
        command.append("--fp16")
    
    if config.flash_attn:
        command.append("--flash_attn")
    
    if config.load_checkpoint:
        command.append("--load_checkpoint")
    
    if config.packing_samples:
        command.append("--packing_samples")
    
    if config.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    
    if config.use_wandb:
        if isinstance(config.use_wandb, bool):
            command.append("--use_wandb")
        else:
            command.append(f"--use_wandb={config.use_wandb}")
        command.append(f"--wandb_project={config.wandb_project}")
    
    if config.deepspeed_config:
        command.append(f"--deepspeed_config={config.deepspeed_config}")
    
    # 执行命令
    import subprocess
    logger.info(f"执行命令: {' '.join(command)}")
    process = subprocess.run(command, check=True)
    
    # 保存配置
    config.to_json(os.path.join(output_dir, "rm_config.json"))
    
    last_ckpt_path = os.path.join(output_dir, "final")
    logger.info(f"奖励模型训练完成，模型保存在: {last_ckpt_path}")
    
    return last_ckpt_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="奖励模型训练")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model_id_or_path", type=str, help="模型ID或路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--dataset", type=str, help="数据集")
    parser.add_argument("--chosen_key", type=str, help="选择文本的字段名")
    parser.add_argument("--rejected_key", type=str, help="拒绝文本的字段名")
    parser.add_argument("--max_length", type=int, help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, help="训练批量大小")
    parser.add_argument("--micro_train_batch_size", type=int, help="微批量大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--max_epochs", type=int, help="最大训练轮数")
    parser.add_argument("--torch_dtype", type=str, choices=["float32", "float16", "bfloat16"], help="模型精度")
    parser.add_argument("--zero_stage", type=int, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO优化阶段")
    parser.add_argument("--deepspeed_config", type=str, help="自定义DeepSpeed配置文件路径")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用Wandb")
    parser.add_argument("--wandb_project", type=str, help="Wandb项目名")
    parser.add_argument("--apply_chat_template", action="store_true", help="是否应用聊天模板")
    parser.add_argument("--flash_attn", action="store_true", help="是否使用Flash Attention")
    parser.add_argument("--load_checkpoint", action="store_true", help="是否加载最新检查点")
    parser.add_argument("--packing_samples", action="store_true", help="是否打包样本")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否使用梯度检查点")
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 创建基础配置
    if args.config:
        # 从配置文件加载
        config = RMConfig.from_json(args.config)
        logger.info(f"从配置文件加载: {args.config}")
    else:
        # 创建默认配置
        config = RMConfig()
    
    # 使用命令行参数覆盖配置
    for arg_name, arg_value in vars(args).items():
        if arg_name != "config" and arg_value is not None:
            setattr(config, arg_name, arg_value)
    
    # 打印配置
    logger.info(f"训练配置: {config.to_dict()}")
    
    # 开始训练
    model_path = train_rm(config)
    logger.info(f"训练完成！模型保存在: {model_path}")


if __name__ == "__main__":
    main()




