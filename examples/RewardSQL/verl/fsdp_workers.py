# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer
import numpy as np
import requests
import itertools
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig, AutoModelForCausalLM
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        def custom_apply_chat_template(chat, add_generation_prompt=True, tokenize=False):
            if isinstance(chat, str):
                return chat
            if isinstance(chat, (list, tuple)):
                return str(chat[0])
            return chat

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))
            self.input_tokenizer.apply_chat_template = custom_apply_chat_template
            self.tokenizer.apply_chat_template = custom_apply_chat_template
        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            
            # 获取 step_tag_id 和 candidate_tokens，可以从配置中获取或作为参数传入
            step_tag_id = self.config.get('step_tag_id', None)
            candidate_tokens = self.config.get('candidate_tokens', None)
            
            if step_tag_id is None or candidate_tokens is None:
                raise ValueError("step_tag_id 和 candidate_tokens 必须在配置中指定")

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                
                logits_rmpad = output.logits  # (1, total_nnz, vocab_size)
                candidate_logits_rmpad = logits_rmpad[:, :, candidate_tokens]  # (1, total_nnz, num_candidates)
                scores_rmpad = torch.nn.functional.softmax(candidate_logits_rmpad, dim=-1)[:, :, 0]  # (1, total_nnz)
                # 将 scores_rmpad 从 (1, total_nnz) 转换为 (total_nnz,)
                scores_rmpad = scores_rmpad.squeeze(0)
                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    scores_rmpad = gather_outpus_and_unpad(scores_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                scores = pad_input(scores_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)  # (batch_size, seqlen)
                
                # 恢复原始 input_ids 以便找到 step_tag_id 的位置
                original_input_ids = pad_input(input_ids_rmpad.transpose(0, 1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
                
                # 获取完整的 logits
                logits = output.logits  # (batch_size, seq_len, vocab_size)
                # 只保留候选 token 的 logits
                candidate_logits = logits[:, :, candidate_tokens]  # (batch_size, seq_len, num_candidates)
                
                # 应用 softmax
                scores = torch.nn.functional.softmax(candidate_logits, dim=-1)[:, :, 0]  # (batch_size, seq_len)
                original_input_ids = input_ids

            # 为每个样本找到所有 step_tag_id 的位置
            batch_scores = []
            for i in range(batch_size):
                # 找到当前样本中所有 step_tag_id 的位置
                step_positions = (original_input_ids[i] == step_tag_id).nonzero(as_tuple=True)[0]
                
                if len(step_positions) == 0:
                    batch_scores.append([0.0])
                else:
                    # 提取这些位置的分数
                    step_scores = scores[i, step_positions]  # 这是一个长度为n的向量
                    steps_scores = step_scores.tolist()  # 将tensor转换为list
                    batch_scores.append(steps_scores)  # 收集每个样本的step分数
            
            # 直接返回列表，不尝试转换为张量
            # rm_score = torch.tensor(batch_scores, dtype=torch.float32, device=scores.device)
            return batch_scores
            
    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores
        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores


    def _expand_to_token_level_steps(self, data: DataProto, scores: list):
        batch_size = data.batch.batch_size[0]
        
        # 防御性代码：检查scores是否有效
        if not scores:
            if self.rank == 0:
                print("警告: scores列表为空，返回全零分数")
            # 如果scores为空，返回全零分数
            attention_mask = data.batch['attention_mask']
            response_length = data.batch['responses'].shape[-1]
            token_level_scores = torch.zeros_like(attention_mask, dtype=torch.float32)
            token_level_scores = token_level_scores[:, -response_length:]
            return token_level_scores
            
        
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=torch.float32)  # (bsz, seqlen)
        
        # 在rank=0下打印出scores情况
        for i in range(batch_size):
            # 检查索引是否越界
            if i >= len(scores):
                if self.rank == 0:
                    print(f"警告：样本索引 {i} 超出scores列表长度 {len(scores)}，跳过")
                continue
                
            # 处理所有类型的scores[i]，统一转换为列表
            if not isinstance(scores[i], list):
                if hasattr(scores[i], 'tolist'):
                    scores[i] = scores[i].tolist()
                else:
                    scores[i] = [float(scores[i])]
                
                if self.rank == 0:
                    print(f"将scores[{i}]转换为列表: {scores[i]}")
            
            # 获取当前样本的响应部分
            response_ids = data.batch['responses'][i]
            valid_response_length = attention_mask[i][-response_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]
            
            if valid_response_length == 0:
                continue
            response_text = self.tokenizer.decode(valid_response_ids)
            newline_positions = []
            current_pos = 0
            while True:
                pos = response_text.find("\n\n", current_pos)
                if pos == -1:
                    break
                newline_positions.append(pos)
                current_pos = pos + 2
            
            # 将分数分配给每个"\n\n"位置和EOS位置
            current_scores = scores[i]
            score_idx = 0
            
            # 直接将分数分配给响应部分的token
            response_scores = torch.zeros(valid_response_length, device=token_level_scores.device)
            
            # 如果没有找到"\n\n"位置，但有分数，则将所有分数分配给EOS位置
            if len(newline_positions) == 0 and len(current_scores) > 0:
                # 将第一个分数分配给最后一个token
                score_value = current_scores[-1]
                response_scores[valid_response_length - 1] = score_value
            else:
                # 为每个"\n\n"分配分数
                for pos in newline_positions:
                    if score_idx < len(current_scores):
                        # 找到对应的token位置
                        token_text = response_text[:pos]
                        token_ids = self.tokenizer.encode(token_text)
                        token_pos = len(token_ids) - 1
                        
                        # 确保token_pos在有效范围内
                        if 0 <= token_pos < valid_response_length:
                            # 列表中的值已经是标量
                            score_value = current_scores[score_idx]
                            response_scores[token_pos] = score_value
                        score_idx += 1
                
                # 最后一个分数分配给最后一个token
                if score_idx < len(current_scores):
                    # 列表中的值已经是标量
                    score_value = current_scores[score_idx]
                    response_scores[valid_response_length - 1] = score_value
            
            # 将response_scores复制到token_level_scores的响应部分
            # 确保valid_response_length > 0，避免空切片错误
            if valid_response_length > 0:
                # 使用正索引而不是负索引，避免空切片问题
                start_idx = token_level_scores.shape[1] - response_length
                end_idx = start_idx + valid_response_length
                token_level_scores[i, start_idx:end_idx] = response_scores
        
        # 打印decode后的response字符和对应的token_level_scores
        if self.rank == 0:  # 只在主进程中打印，避免重复输出
            for i in range(min(1, batch_size)):  # 只打印前3个样本，避免输出过多
                response_ids = data.batch['responses'][i]
                valid_response_length = attention_mask[i][-response_length:].sum().item()
                
                # 如果有效响应长度为0，则跳过
                if valid_response_length == 0:
                    print(f"\n样本 {i} 的响应长度为0，跳过")
                    continue
                
                valid_response_ids = response_ids[:valid_response_length]
                
                # 解码响应文本
                response_text = self.tokenizer.decode(valid_response_ids)
                
                # 获取该样本的token级别分数
                start_idx = token_level_scores.shape[1] - response_length
                end_idx = start_idx + valid_response_length
                sample_scores = token_level_scores[i, start_idx:end_idx]
                
                print(f"\n样本 {i} 的响应:")
                # print(f"响应文本: {response_text}")
                print(f"Token级别分数非零元素数量: {torch.sum(sample_scores != 0).item()}")
                
                # 如果需要更详细的token级别分析
                tokens = self.tokenizer.convert_ids_to_tokens(valid_response_ids)
                non_zero_indices = torch.nonzero(sample_scores).squeeze(-1).tolist()
                if isinstance(non_zero_indices, int):
                    non_zero_indices = [non_zero_indices]
                if non_zero_indices:
                    print("非零分数的token位置:")
                    for idx in non_zero_indices:
                        token_idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                        if token_idx < len(tokens):
                            print(f"位置 {token_idx}, Token: {tokens[token_idx]}, 分数: {sample_scores[token_idx].item()}")
        
        # 如果所有token_level_scores都是0，则至少给每个样本的EOS位置分配一个默认分数
        # 检查每个位置是否都为0，而不是简单地检查总和
        if (token_level_scores == 0).all():
            if self.rank == 0:
                print("警告：所有token_level_scores都是0，将为每个样本的最后一个有效位置分配默认分数0.01")
            for i in range(batch_size):
                # 使用最后一个有效位置作为EOS位置
                valid_response_length = attention_mask[i][-response_length:].sum().item()
                if valid_response_length > 0:
                    # 使用正索引而不是负索引
                    start_idx = token_level_scores.shape[1] - response_length
                    token_level_scores[i, start_idx + valid_response_length - 1] = 0.01
        
        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores
    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: str = data.non_tensor_batch['raw_prompt'][i]
            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')
            if "\n\n" in response:
                response = response.replace("\n\n", " и ") + " и "
            # 替换直接调用process_single_item为HTTP请求
            if self.config.enable_EXreward:
                try:
                    api_url = f"http://localhost:{self.config.port}/process_item"
                    payload = {
                        "db_id": data.non_tensor_batch['db_id'][i],
                        "question_id": data.non_tensor_batch['index'][i],
                        "sql": response
                    }
                    
                    # 发送HTTP请求到Flask API
                    api_response = requests.post(api_url, json=payload, timeout=120)
                    
                    if api_response.status_code == 200:
                        response_data = api_response.json()
                        if response_data['status'] == 'success':
                            response = response_data['result']
                        else:
                            response = response
                    else:
                        response = response
                except Exception as e:
                    response = response
            if self.rank == 0 and i == 0:
                print(f"Switch template. response: {response}")
            chat += response

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_scores = self._forward_micro_batch(micro_batch) 
                output.append(rm_scores)


            scores = output 
            if use_dynamic_bsz:
                # 扁平化indices列表
                flat_indices = list(itertools.chain.from_iterable(indices))
                # 将嵌套列表扁平化为一维列表
                flat_scores = []
                for batch_scores in scores:
                    flat_scores.extend(batch_scores)
                # 检查数量是否匹配
                assert len(flat_indices) == len(flat_scores), f"{len(flat_indices)} vs. {len(flat_scores)}"
                # 获取原始顺序的索引
                revert_indices = get_reverse_idx(flat_indices)
                
                # 根据revert_indices重新排列scores
                sorted_scores = []
                for idx in revert_indices:
                    sorted_scores.append(flat_scores[idx])
                
                scores = sorted_scores
            else:
                flat_scores = []
                for batch_scores in scores:
                    flat_scores.extend(batch_scores)
                scores = flat_scores

            # 修改_expand_to_token_level函数以处理多个分数
            token_level_scores = self._expand_to_token_level_steps(data, scores)
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
