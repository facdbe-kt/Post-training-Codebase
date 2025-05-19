import os
import json
import argparse
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference import Utils
from sql_decoders import SQLBestOfN

def load_dataset(dataset_path):
    """加载数据集"""
    with open(dataset_path, "r") as file:
        return json.load(file)

def load_db_ids(db_ids_path):
    """加载数据库ID"""
    with open(db_ids_path, "r") as file:
        dbs = json.load(file)
        return [item["db_id"] for item in dbs]

def inference(test_set, dbs, args):
    """执行SQL推理"""
    # 创建SQL优化的Best-of-N解码器
    decoder = SQLBestOfN(
        model_name_or_path=args.model,
        hyperparameters={
            "model_type": args.type,
            "max_tokens": args.max_tokens,
            "temperature": 0.7,  # 使用较高的温度以增加多样性
            "top_p": 0.9,
            "n": args.n_candidates,  # 生成多个候选
            "stop": ["\n\n"],
            # OpenAI特定参数
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "api_base": os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
            # VLLM特定参数
            "gpu_memory_utilization": 0.95,
            "dtype": "float16"
        }
    )
    
    # 结果存储
    results = []
    
    # 使用tqdm显示进度
    for index, item in enumerate(tqdm(test_set)):
        # 执行推理
        sql, metadata = decoder.forward(item["instruction"])
        
        if index < 2:
            print(f"指令: {item['instruction']}")
            print(f"生成的SQL: {sql}")
            print(f"候选数量: {len(metadata['candidates'])}")
            print(f"最佳候选索引: {metadata['best_candidate_index']}")
            print(f"评分: {metadata['scores']}")
            print()
        
        # 处理输出
        if args.mode == "cte_rationale":
            # 提取SQL代码块
            sub_sqls = Utils.extract_sql_from_str(sql)
            final_sql = ""
            for sub_sql in sub_sqls:
                final_sql += sub_sql + "\n\n"
        else:
            final_sql = sql
        
        # 清理特殊字符
        if " \u0438" in final_sql:
            final_sql = final_sql.replace(" \u0438", " ")
        
        # 保存结果
        results.append({
            "question_id": index, 
            "instruction": item["instruction"],
            "output": item["output"],
            "final_answer": final_sql,
            "raw_output": sql,
            "all_candidates": metadata["candidates"],
            "scores": metadata["scores"]
        })
    
    # 格式化输出结果
    bird_results = {}
    for index, item in enumerate(results):
        bird_results[str(index)] = item["final_answer"] + "\t----- bird -----\t" + dbs[index]
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 保存结果
    with open(os.path.join(args.output, "predict_dev.json"), "w") as file:
        json.dump(bird_results, file, indent=4)
    
    # 保存完整结果（包括所有候选）
    with open(os.path.join(args.output, "all_candidates.json"), "w") as file:
        json.dump(results, file, indent=4)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用Best-of-N解码策略进行SQL推理")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件的保存路径")
    parser.add_argument("--type", type=str, choices=["openai", "vllm", "huggingface"], default="vllm", help="模型类型")
    parser.add_argument("--model", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--mode", type=str, choices=["cte", "cte_rationale", "origin"], default="origin", help="SQL生成模式")
    parser.add_argument("--dataset", type=str, default="/fs/fast/u2020201469/models/saves/data/m_schema/mschema_prompt_dev_bird.json", help="数据集路径")
    parser.add_argument("--db_ids", type=str, default="/home/u2020201469/NL2SQL/openr/datasets/bird/dev.json", help="数据库ID文件")
    parser.add_argument("--max_tokens", type=int, default=256, help="生成的最大token数")
    parser.add_argument("--n_candidates", type=int, default=5, help="生成的候选数量")
    
    args = parser.parse_args()
    
    # 加载数据
    test_set = load_dataset(args.dataset)
    dbs = load_db_ids(args.db_ids)
    
    # 执行推理
    inference(test_set, dbs, args)

if __name__ == "__main__":
    main() 