#!/usr/bin/env python3
"""
VERL训练示例脚本

展示如何使用训练管理器进行VERL训练
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.trainer.training_manager import (
    VERLTrainingManager, 
    register_reward_manager, 
    register_reward_function,
    register_core_algos
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入自定义奖励管理器
from examples.RewardSQL.verl.custom import CustomRewardManager, reward_func
# 导入自定义核心算法
import examples.RewardSQL.verl.core_algos


# 注册奖励管理器和函数
@register_reward_manager("sql_reward_manager")
class SQLRewardManager(CustomRewardManager):
    """SQL奖励管理器"""
    pass


@register_reward_function("sql_reward_func")
def sql_reward_function(*args, **kwargs):
    """SQL奖励函数"""
    return reward_func(*args, **kwargs)


# 注册核心算法模块
@register_core_algos("sql_core_algos")
class SQLCoreAlgos:
    """SQL核心算法封装"""
    
    @staticmethod
    def compute_grpo_process_advantage_avg_unique(*args, **kwargs):
        """封装compute_grpo_process_advantage_avg_unique算法"""
        return examples.RewardSQL.verl.core_algos.compute_grpo_process_advantage_avg_unique(*args, **kwargs)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VERL训练示例")
    parser.add_argument("--config", type=str, default="examples/RewardSQL/verl/grpo_trainer.yaml", 
                        help="配置文件路径")
    parser.add_argument("--algorithm", type=str, default="grpo", 
                        choices=["grpo", "ppo", "prime", "dapo"], 
                        help="训练算法")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="outputs/verl_train", 
                        help="输出目录")
    parser.add_argument("--core_algos", type=str, default="examples.RewardSQL.verl.core_algos", 
                        help="核心算法模块路径")
    parser.add_argument("--core_function", type=str, default=None,
                        help="核心算法函数名称，默认自动选择")
    parser.add_argument("--reward_manager", type=str, default="sql_reward_manager",
                        help="奖励管理器名称")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="训练轮数")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化训练管理器
    manager = VERLTrainingManager(
        config=args.config,
        algorithm_type=args.algorithm,
        core_algos_module=args.core_algos,
        core_algos_function=args.core_function
    )
    
    # 执行训练
    try:
        # 准备额外参数
        kwargs = {}
        if args.epochs is not None:
            kwargs["num_epochs"] = args.epochs
        
        # 添加奖励管理器
        if args.reward_manager:
            kwargs["reward_manager"] = args.reward_manager
            
        # 开始训练
        logger.info(f"开始使用{args.algorithm}算法训练，核心函数: {args.core_function or '自动选择'}")
        stats = manager.train(
            data_path=args.data_path,
            output_dir=args.output_dir,
            **kwargs
        )
        
        # 打印训练结果
        logger.info(f"训练完成，结果: {stats}")
        
        # 保存最终模型
        save_path = manager.save(args.output_dir)
        logger.info(f"最终模型已保存到: {save_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    main() 