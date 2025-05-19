# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


def compute_grpo_process_advantage_avg_unique(token_level_rewards: torch.Tensor,
                                  eos_mask: torch.Tensor,
                                  index: torch.Tensor,
                                  epsilon: float = 1e-6):
    """
    计算基于过程监督的GRPO优势函数，使用每个response中不重复分数的平均值作为归一化基准。
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            形状: (bs, response_length)
            其中不为0的位置表示步骤结束处的奖励值
        eos_mask: `(torch.Tensor)`
            形状: (bs, response_length)
        index: `(torch.Tensor)`
            形状: (bs,)，表示每个样本对应的提示索引
            
    Returns:
        advantages: `(torch.Tensor)`
            形状: (bs, response_length)，每个标记的优势值
        Returns: `(torch.Tensor)`
            形状: (bs, response_length)，与优势值相同
    """
    bsz, response_length = token_level_rewards.shape
    
    # 创建优势张量
    advantages = torch.zeros_like(token_level_rewards)
    
    with torch.no_grad():
        # 按照提示索引分组
        prompt_indices = defaultdict(list)
        for i in range(bsz):
            prompt_indices[index[i]].append(i)
        
        # 对每个提示组分别处理
        for prompt_idx, sample_indices in prompt_indices.items():
            # 第一步：计算每个response中不重复分数的平均值
            avg_rewards = []
            sample_to_avg_reward = {}
            
            for i in sample_indices:
                rewards = token_level_rewards[i]
                # 找出非零奖励的位置和值
                non_zero_positions = torch.nonzero(rewards, as_tuple=True)[0]
                
                if len(non_zero_positions) > 0:
                    # 获取所有非零奖励值
                    non_zero_rewards = rewards[non_zero_positions]
                    # 获取不重复的奖励值
                    unique_rewards = torch.unique(non_zero_rewards)
                    # 计算不重复奖励值的平均值
                    avg_reward = unique_rewards.mean().item()
                else:
                    avg_reward = 0.0
                
                avg_rewards.append(avg_reward)
                sample_to_avg_reward[i] = avg_reward
            
            # 如果该提示组只有一个样本，则优势值为0
            if len(avg_rewards) <= 1:
                for i in sample_indices:
                    advantages[i] = torch.zeros_like(token_level_rewards[i])
                continue
            
            # 检查是否所有回答分数都在1之下或者1之上
            all_below_one = all(avg_reward < 1.0 for avg_reward in avg_rewards)
            all_above_one = all(avg_reward > 1.0 for avg_reward in avg_rewards)
            
            # 如果全对或全错，则将该组的所有优势值设为0
            # if all_below_one or all_above_one:
            #     for i in sample_indices:
            #         advantages[i] = torch.zeros_like(token_level_rewards[i])
            #     continue
            
            # 第二步：对平均奖励值进行标准化，计算每个response的优势值
            avg_rewards_tensor = torch.tensor(avg_rewards)
            mean_reward = avg_rewards_tensor.mean().item()
            std_reward = avg_rewards_tensor.std().item() + epsilon
            
            # 计算每个response的标准化优势值
            sample_to_advantage = {}
            for i in sample_indices:
                normalized_advantage = (sample_to_avg_reward[i] - mean_reward) / std_reward
                sample_to_advantage[i] = normalized_advantage
            
            # 第三步：根据每个response的优势值和每个step的reward分配优势
            for i in sample_indices:
                rewards = token_level_rewards[i]
                response_advantage = sample_to_advantage[i]
                # 找出非零奖励的位置和值
                non_zero_positions = torch.nonzero(rewards, as_tuple=True)[0]
                
                # 对每个非零位置应用计算advantage规则
                for pos in non_zero_positions:
                    reward = rewards[pos].item()
                    # 根据response优势值的正负决定分配方式
                    if response_advantage > 0:
                        # 对于好的样本，直接用step reward乘以response advantage
                        if reward > 1.0:
                            step_advantage = (reward - 1.0) * response_advantage
                        else:
                            step_advantage = reward * response_advantage
                    else:
                        # 对于差的样本，用(1-reward)乘以response advantage
                        step_advantage = (1.0 - reward) * response_advantage
                    
                    # 将计算的优势值应用到对应位置
                    advantages[i, pos] = step_advantage
    
    return advantages, advantages


def compute_weighted_grpo_advantage(token_level_rewards: torch.Tensor,
                                  eos_mask: torch.Tensor,
                                  index: torch.Tensor,
                                  weight_factor: float = 0.5,
                                  epsilon: float = 1e-6):
    """
    计算带权重的GRPO优势函数，平衡批次内样本的影响
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            形状: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            形状: (bs, response_length)
        index: `(torch.Tensor)`
            形状: (bs,)，表示每个样本对应的提示索引
        weight_factor: 权重因子，控制样本间平衡度
            
    Returns:
        advantages: `(torch.Tensor)`
            形状: (bs, response_length)
        Returns: `(torch.Tensor)`
            形状: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2weight = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 按提示索引分组
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            
        # 计算每组的统计量
        for idx in id2score:
            scores_list = id2score[idx]
            if len(scores_list) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
                id2weight[idx] = torch.tensor(1.0)
            elif len(scores_list) > 1:
                scores_tensor = torch.tensor(scores_list)
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
                
                # 计算样本权重
                max_score = torch.max(scores_tensor)
                min_score = torch.min(scores_tensor)
                range_score = max_score - min_score
                
                # 防止除零
                if range_score == 0:
                    weights = torch.ones_like(scores_tensor)
                else:
                    # 将分数归一化到[0,1]区间
                    normalized_scores = (scores_tensor - min_score) / range_score
                    # 加权，使较小的分数也有一定权重
                    weights = weight_factor + (1 - weight_factor) * normalized_scores
                
                id2weight[idx] = weights
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # 计算带权重的优势
        advantages = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            prompt_idx = index[i]
            score_idx = id2score[prompt_idx].index(scores[i])
            weight = id2weight[prompt_idx][score_idx]
            
            # 应用归一化和权重
            if len(id2score[prompt_idx]) > 1:
                normalized_adv = (scores[i] - id2mean[prompt_idx]) / (id2std[prompt_idx] + epsilon)
                weighted_adv = normalized_adv * weight
                advantages[i] = weighted_adv.unsqueeze(-1) * eos_mask[i]
    
    return advantages, advantages


def compute_token_wise_advantage(token_level_rewards: torch.Tensor,
                             eos_mask: torch.Tensor,
                             index: torch.Tensor,
                             epsilon: float = 1e-6):
    """
    计算token级别的优势函数，不仅考虑总体reward，还考虑每个token位置的分布
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            形状: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            形状: (bs, response_length)
        index: `(torch.Tensor)`
            形状: (bs,)，表示每个样本对应的提示索引
            
    Returns:
        advantages: `(torch.Tensor)`
            形状: (bs, response_length)
        Returns: `(torch.Tensor)`
            形状: (bs, response_length)
    """
    bsz, response_length = token_level_rewards.shape
    
    # 按提示组织数据
    prompt_to_samples = defaultdict(list)
    for i in range(bsz):
        prompt_to_samples[index[i].item()].append(i)
    
    # 创建优势张量
    advantages = torch.zeros_like(token_level_rewards)
    
    with torch.no_grad():
        # 对每个提示组分别处理
        for prompt_idx, sample_indices in prompt_to_samples.items():
            # 跳过只有一个样本的组
            if len(sample_indices) <= 1:
                continue
                
            # 为每个token位置计算统计量
            for pos in range(response_length):
                pos_rewards = []
                
                # 收集该位置所有样本的奖励值
                for i in sample_indices:
                    if eos_mask[i, pos] > 0:  # 只考虑有效位置
                        pos_rewards.append(token_level_rewards[i, pos].item())
                
                # 如果该位置没有足够样本，跳过
                if len(pos_rewards) <= 1:
                    continue
                    
                # 计算该位置的统计量
                pos_rewards = torch.tensor(pos_rewards)
                pos_mean = torch.mean(pos_rewards)
                pos_std = torch.std(pos_rewards) + epsilon
                
                # 为每个样本计算该位置的优势值
                for i in sample_indices:
                    if eos_mask[i, pos] > 0:
                        normalized_adv = (token_level_rewards[i, pos] - pos_mean) / pos_std
                        advantages[i, pos] = normalized_adv
    
    return advantages, advantages



