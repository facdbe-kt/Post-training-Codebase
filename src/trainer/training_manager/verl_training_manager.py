"""
VERL训练管理器

封装VERL框架的功能，提供简单的接口来使用VERL进行模型训练
"""

import os
import yaml
import logging
import tempfile
from typing import Dict, Any, Optional, List, Union, Callable, Type
import importlib
from pathlib import Path
import copy
import torch
import sys

from .rl_training_manager import RLTrainingManager
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

# VERL相关导入
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../verl')))
    from verl.single_controller.base import Worker
    VERL_AVAILABLE = True
except ImportError:
    logger.warning("VERL框架导入失败，可能需要安装或修复VERL")
    VERL_AVAILABLE = False


class VERLTrainingManager(RLTrainingManager):
    """
    VERL训练管理器
    
    封装VERL框架的功能，简化VERL的使用
    """
    
    SUPPORTED_ALGORITHMS = ["ppo", "grpo", "dapo", "prime", "r1", "drgrpo"]
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 algorithm_type: str = "grpo",
                 reward_function: Optional[Callable] = None,
                 reward_manager_class: Optional[Type] = None,
                 core_algos_module: Optional[str] = None,
                 core_algos_function: Optional[str] = None):
        """
        初始化VERL训练管理器
        
        Args:
            config: 训练配置，可以是字典或YAML文件路径
            algorithm_type: 算法类型，支持 "ppo", "grpo", "dapo", "prime" 等
            reward_function: 可选的奖励函数
            reward_manager_class: 自定义奖励管理器类
            core_algos_module: 自定义算法模块路径，如果不提供，将使用默认模块
            core_algos_function: 自定义算法函数名，用于替换默认的优势计算函数
        """
        if not VERL_AVAILABLE:
            raise ImportError("VERL框架未安装或导入失败，无法使用VERLTrainingManager")
            
        # 处理配置，可以是字典、YAML文件路径或OmegaConf对象
        if isinstance(config, str) and os.path.exists(config):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.verl_config = OmegaConf.create(config_dict)
        elif isinstance(config, Dict):
            self.verl_config = OmegaConf.create(config)
        elif isinstance(config, DictConfig):
            self.verl_config = config
        else:
            raise ValueError(f"配置类型不支持: {type(config)}")
        
        # 初始化基类
        super().__init__(OmegaConf.to_container(self.verl_config, resolve=True))
        
        # 保存参数
        self.algorithm_type = algorithm_type.lower()
        if self.algorithm_type not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"不支持的算法类型: {algorithm_type}，支持的类型: {self.SUPPORTED_ALGORITHMS}")
        
        self.reward_function = reward_function
        self.reward_manager_class = reward_manager_class
        self.core_algos_module = core_algos_module
        self.core_algos_function = core_algos_function
        
        # VERL控制器、工作器等组件
        self.controller = None
        self.actor_worker = None
        self.critic_worker = None
        self.reward_worker = None
        self.trainer = None
        
        # 确认算法配置
        self._override_algorithm_config()
    
    def _override_algorithm_config(self):
        """根据选择的算法类型覆盖配置"""
        # 更新算法类型
        with OmegaConf.open_dict(self.verl_config):
            self.verl_config.algorithm.adv_estimator = self.algorithm_type
    
    def initialize(self) -> None:
        """初始化VERL训练所需的资源和模型"""
        if self.initialized:
            logger.warning("训练管理器已经初始化，跳过重复初始化")
            return
            
        logger.info(f"初始化VERL训练环境，算法: {self.algorithm_type}...")
        
        self._initialize_controller()
        self._initialize_workers()
        self._initialize_core_module()
        
        self.initialized = True
        logger.info("VERL训练管理器初始化完成")
    
    def _initialize_controller(self):
        """初始化VERL控制器"""
        try:
            from verl.single_controller.controller import Controller
            self.controller = Controller()
        except ImportError as e:
            raise ImportError(f"初始化VERL控制器失败: {e}")
    
    def _initialize_workers(self):
        """初始化VERL工作器"""
        try:
            # 导入工作器类
            from verl.single_controller.ppo import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
            
            # 根据配置创建工作器
            self.actor_worker = self.controller.new_worker(
                ActorRolloutRefWorker, self.verl_config, "actor_rollout_ref"
            )
            
            if self.algorithm_type not in ["grpo", "drgrpo"]:  # GRPO不需要critic
                self.critic_worker = self.controller.new_worker(
                    CriticWorker, self.verl_config
                )
            
            if self.verl_config.reward_model.enable:
                self.reward_worker = self.controller.new_worker(
                    RewardModelWorker, self.verl_config
                )
                
            # 初始化模型
            self.actor_worker.init_model()
            if self.critic_worker:
                self.critic_worker.init_model()
            if self.reward_worker:
                self.reward_worker.init_model()
                
        except Exception as e:
            raise RuntimeError(f"初始化VERL工作器失败: {e}")
    
    def _initialize_core_module(self):
        """初始化核心算法模块"""
        try:
            # 如果提供了自定义算法模块，则导入
            if self.core_algos_module:
                if os.path.isfile(self.core_algos_module):
                    module_path = os.path.abspath(self.core_algos_module)
                    module_dir = os.path.dirname(module_path)
                    sys.path.insert(0, module_dir)
                    module_name = os.path.splitext(os.path.basename(module_path))[0]
                    self.core_module = importlib.import_module(module_name)
                else:
                    self.core_module = importlib.import_module(self.core_algos_module)
                
                # 注册核心算法模块
                from .component_registry import registry
                registry.register("core_algos", "custom_core_algos", self.core_module)
                
                # 修补VERL计算优势函数的方法
                self._patch_verl_advantage_computation()
            else:
                # 使用默认模块
                trainer_module_name = f"verl.trainer.ppo.core_algos"
                self.core_module = importlib.import_module(trainer_module_name)
                
        except Exception as e:
            raise ImportError(f"导入核心算法模块失败: {e}")
            
    def _patch_verl_advantage_computation(self):
        """
        修补VERL计算优势函数的方法，使用自定义算法替换原有算法
        
        根据配置动态选择要使用的计算优势函数
        """
        try:
            import verl.trainer.ppo.ray_trainer as ray_trainer
            import verl.trainer.ppo.core_algos as core_algos
            
            # 替换原有的计算逻辑
            original_compute_advantage = ray_trainer.compute_advantage
            
            # 获取要使用的自定义函数
            custom_function = None
            function_name = self.core_algos_function
            
            # 如果未在初始化时指定函数名，则尝试从配置文件中获取
            if not function_name and hasattr(self.verl_config, "algorithm") and hasattr(self.verl_config.algorithm, "core_function"):
                function_name = self.verl_config.algorithm.core_function
                logger.info(f"从配置文件中获取核心算法函数名: {function_name}")
            
            # 如果未指定函数名，则搜索可能的函数
            if not function_name:
                # 根据算法类型寻找相应函数
                if self.algorithm_type == "grpo":
                    candidate_names = [
                        "compute_grpo_process_advantage_avg_unique",
                        "compute_grpo_advantage",
                        "compute_weighted_grpo_advantage",
                        "compute_token_wise_advantage"
                    ]
                    
                    # 按优先级顺序查找
                    for name in candidate_names:
                        if hasattr(self.core_module, name):
                            function_name = name
                            break
            
            # 如果找到函数名，获取函数对象
            if function_name and hasattr(self.core_module, function_name):
                custom_function = getattr(self.core_module, function_name)
                logger.info(f"找到自定义核心算法函数: {function_name}")
            else:
                logger.warning(f"未找到自定义核心算法函数，将使用默认算法")
                return
            
            # 创建新的计算优势函数
            def patched_compute_advantage(data, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
                # 对GRPO算法使用自定义优势函数
                if adv_estimator == ray_trainer.AdvantageEstimator.GRPO and custom_function:
                    grpo_calculation_mask = data.batch["response_mask"]
                    if multi_turn:
                        # 如果是多轮对话，则使用loss_mask的相关部分
                        response_length = grpo_calculation_mask.size(1)
                        grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
                    
                    # 使用自定义优势函数计算
                    advantages, returns = custom_function(
                        token_level_rewards=data.batch["token_level_rewards"],
                        eos_mask=grpo_calculation_mask,
                        index=data.non_tensor_batch["uid"],
                    )
                    data.batch["advantages"] = advantages
                    data.batch["returns"] = returns
                    return data
                
                # 其他算法使用原始函数
                return original_compute_advantage(data, adv_estimator, gamma, lam, num_repeat, multi_turn, norm_adv_by_std_in_grpo)
            
            # 替换原有函数
            ray_trainer.compute_advantage = patched_compute_advantage
            
            logger.info(f"成功修补VERL计算优势函数，使用自定义算法: {function_name}")
            
        except Exception as e:
            logger.warning(f"修补VERL计算优势函数失败: {e}，将使用默认算法")
        
    def _setup_reward_function(self):
        """设置奖励函数"""
        # 已在初始化时处理，VERL使用独特的奖励管理机制
        pass
        
    def _initialize_models(self):
        """初始化模型"""
        # 已在_initialize_workers中处理
        pass
        
    def _initialize_optimizers(self):
        """初始化优化器"""
        # 已在_initialize_workers中处理
        pass
        
    def _setup_reward_manager(self, tokenizer=None):
        """设置奖励管理器"""
        if not self.reward_manager_class:
            # 使用配置中指定的默认管理器
            logger.info("使用默认奖励管理器")
            return None
            
        # 使用自定义奖励管理器
        try:
            if tokenizer is None and hasattr(self.actor_worker, "tokenizer"):
                tokenizer = self.actor_worker.tokenizer
                
            reward_manager = self.reward_manager_class(
                tokenizer=tokenizer,
                num_examine=self.verl_config.get("debug_print_num", 2),
                config=self.verl_config,
                compute_score=self.reward_function
            )
            logger.info(f"已创建自定义奖励管理器: {self.reward_manager_class.__name__}")
            return reward_manager
        except Exception as e:
            logger.error(f"创建自定义奖励管理器失败: {e}")
            return None
        
    def train(self, 
              data_path: Union[str, List[str]], 
              output_dir: str, 
              **kwargs) -> Dict[str, Any]:
        """
        执行VERL训练
        
        Args:
            data_path: 训练数据路径，可以是单个文件路径或多个文件路径的列表
            output_dir: 输出目录路径
            **kwargs: 其他训练参数，会覆盖配置文件中的设置
                - num_epochs: 训练轮数
                - batch_size: 批大小
                - learning_rate: 学习率
                - eval_interval: 评估间隔
                - save_interval: 保存间隔
                - reward_manager: 奖励管理器名称或实例
                
        Returns:
            训练统计信息的字典
        """
        if not self.initialized:
            self.initialize()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理数据路径
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path
            
        # 更新训练配置
        with OmegaConf.open_dict(self.verl_config):
            # 处理数据配置
            if "data" not in self.verl_config:
                self.verl_config.data = {}
            self.verl_config.data.train_file = data_paths
            
            # 处理输出目录
            if "trainer" not in self.verl_config:
                self.verl_config.trainer = {}
            self.verl_config.trainer.output_dir = output_dir
            
            # 处理其他训练参数
            for key, value in kwargs.items():
                if key == "num_epochs" and value is not None:
                    self.verl_config.trainer.num_epochs = value
                elif key == "batch_size" and value is not None:
                    self.verl_config.data.batch_size = value
                elif key == "learning_rate" and value is not None:
                    self.verl_config.actor_rollout_ref.actor.learning_rate = value
                elif key == "eval_interval" and value is not None:
                    self.verl_config.trainer.eval_interval = value
                elif key == "save_interval" and value is not None:
                    self.verl_config.trainer.save_interval = value
        
        try:
            # 导入训练器模块
            if self.algorithm_type == "ppo":
                from verl.trainer.ppo.ray_trainer import RayPPOTrainer as Trainer
            elif self.algorithm_type in ["grpo", "drgrpo"]:
                from verl.trainer.ppo.ray_trainer import RayPPOTrainer as Trainer
            elif self.algorithm_type == "prime":
                from verl.recipe.prime.ray_trainer import PrimeRayTrainer as Trainer
            elif self.algorithm_type == "dapo":
                from verl.recipe.dapo.ray_trainer import DAPORayTrainer as Trainer
            else:
                raise ValueError(f"不支持的算法类型: {self.algorithm_type}")
                
            # 设置奖励管理器
            reward_manager = kwargs.get("reward_manager")
            if reward_manager:
                if isinstance(reward_manager, str):
                    # 从注册表中获取
                    from .component_registry import registry
                    reward_manager_class = registry.get_reward_manager(reward_manager)
                    if not reward_manager_class:
                        raise ValueError(f"找不到奖励管理器: {reward_manager}")
                    self.reward_manager_class = reward_manager_class
            
            # 实例化训练器
            if self.reward_manager_class:
                # 如果设置了自定义奖励管理器，在这里初始化
                tokenizer = getattr(self.actor_worker, "tokenizer", None)
                custom_reward_manager = self._setup_reward_manager(tokenizer)
                
                # 创建训练器时传入自定义奖励管理器
                trainer = Trainer(
                    config=self.verl_config,
                    controller=self.controller,
                    actor_rollout_wg=self.actor_worker,
                    critic_wg=self.critic_worker,
                    reward_model_wg=self.reward_worker,
                    reward_manager=custom_reward_manager
                )
            else:
                # 使用默认奖励管理器
                trainer = Trainer(
                    config=self.verl_config,
                    controller=self.controller,
                    actor_rollout_wg=self.actor_worker,
                    critic_wg=self.critic_worker,
                    reward_model_wg=self.reward_worker
                )
                
            # 保存训练器实例
            self.trainer = trainer
            
            # 执行训练
            logger.info(f"开始执行VERL训练，算法: {self.algorithm_type}, 数据: {data_paths}")
            stats = trainer.fit()
            
            # 保存最终模型
            if "save_final_model" not in kwargs or kwargs["save_final_model"]:
                self.save(output_dir)
                
            return stats
            
        except Exception as e:
            logger.error(f"VERL训练失败: {e}")
            raise
            
    def evaluate(self, 
                 data_path: Union[str, List[str]], 
                 **kwargs) -> Dict[str, Any]:
        """
        评估当前策略
        
        Args:
            data_path: 评估数据路径
            **kwargs: 额外参数
                
        Returns:
            评估结果统计信息
        """
        if not self.initialized:
            self.initialize()
            
        # 处理评估数据路径
        if isinstance(data_path, str):
            val_files = data_path
        elif isinstance(data_path, list):
            val_files = ",".join(data_path)
        else:
            raise ValueError(f"不支持的数据路径类型: {type(data_path)}")
            
        # 更新配置
        with OmegaConf.open_dict(self.verl_config):
            self.verl_config.data.val_files = val_files
            
        # 执行评估
        if not hasattr(self, 'trainer') or self.trainer is None:
            raise RuntimeError("请先调用train()方法初始化训练器")
            
        try:
            eval_stats = self.trainer.evaluate()
            return eval_stats
        except Exception as e:
            logger.error(f"VERL评估失败: {e}")
            raise
            
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
        if not self.initialized or not hasattr(self, 'actor_worker'):
            raise RuntimeError("训练管理器尚未初始化，无法保存")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取步骤信息
        step_info = f"step_{step}" if step is not None else "final"
        save_path = os.path.join(output_dir, step_info)
        
        # 保存模型
        try:
            self.actor_worker.save_checkpoint(save_path)
            if hasattr(self, 'critic_worker') and self.critic_worker is not None:
                critic_path = os.path.join(save_path, "critic")
                os.makedirs(critic_path, exist_ok=True)
                self.critic_worker.save_checkpoint(critic_path)
                
            logger.info(f"模型已保存到: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
            
    def load(self, model_path: str) -> None:
        """
        加载训练状态和模型
        
        Args:
            model_path: 模型路径
        """
        if not self.initialized:
            self.initialize()
            
        try:
            # 加载模型
            self.actor_worker.load_checkpoint(model_path)
            
            # 检查是否有critic模型
            critic_path = os.path.join(model_path, "critic")
            if os.path.exists(critic_path) and hasattr(self, 'critic_worker') and self.critic_worker is not None:
                self.critic_worker.load_checkpoint(critic_path)
                
            logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
            
    def generate(self, 
                 prompts: List[str], 
                 **kwargs) -> List[str]:
        """
        使用当前策略生成回复
        
        Args:
            prompts: 输入提示列表
            **kwargs: 生成参数
                - temperature: 温度参数，默认为1.0
                - top_p: top-p采样参数，默认为1.0
                - max_length: 最大生成长度
                
        Returns:
            生成的回复列表
        """
        if not self.initialized:
            self.initialize()
            
        # 创建临时数据集
        from verl import DataProto
        import torch
        
        # 将提示转换为DataProto对象
        data = DataProto()
        data.non_tensor_batch = {
            "raw_text": prompts
        }
        
        # 设置生成参数
        generation_config = {
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "max_length": kwargs.get("max_length", self.verl_config.data.max_response_length)
        }
        
        # 调用worker生成回复
        try:
            results = self.actor_worker.generate_sequences(data, **generation_config)
            responses = results.non_tensor_batch.get("generated_text", [])
            
            if not responses and "raw_responses" in results.non_tensor_batch:
                # 可能需要手动解码
                tokenizer = getattr(self.actor_worker, "tokenizer", None)
                if tokenizer:
                    raw_responses = results.non_tensor_batch["raw_responses"]
                    responses = [tokenizer.decode(resp) for resp in raw_responses]
                    
            return responses
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            raise 