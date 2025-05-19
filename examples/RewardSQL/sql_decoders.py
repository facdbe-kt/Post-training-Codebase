import sys
import os
from typing import Dict, Any, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.best_of_n import BestOfN 
from src.inference.self_improve import SelfImprove

class SQLBestOfN(BestOfN):
    """SQL任务特定的Best-of-N解码策略类"""
    
    def score(self, candidate: str, context: Dict[str, Any] = None) -> float:
        """
        对SQL候选输出进行评分
        
        Args:
            candidate: 候选SQL语句
            context: 评分上下文信息
            
        Returns:
            float: 候选SQL语句的得分，分数越高表示质量越好
        """
        score = 0
        
        # 基本分数：长度
        # 一般来说，SQL查询不会过长，但太短的查询可能不完整
        length = len(candidate)
        if 10 <= length <= 500:
            score += 10
        else:
            score -= (abs(length - 250) / 50)  # 惩罚过长或过短的查询
        
        # 语法检查：关键字计数
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"]
        keyword_count = sum(1 for keyword in sql_keywords if keyword.upper() in candidate.upper())
        score += keyword_count * 2
        
        # 检查SQL语句是否完整
        if "SELECT" in candidate.upper() and "FROM" in candidate.upper():
            score += 20
        
        # 检查SQL语法错误
        error_indicators = ["SELCT", "FORM", "WHER", "ORDR BY"]  # 常见拼写错误
        for indicator in error_indicators:
            if indicator in candidate.upper():
                score -= 30
        
        # 检查SQL注入风险
        injection_indicators = ["DROP", "DELETE FROM", "UPDATE", "INSERT INTO"]
        for indicator in injection_indicators:
            if indicator in candidate.upper():
                score -= 50
        
        return score


class SQLSelfImprove(SelfImprove):
    """SQL任务特定的自我改进解码策略类"""
    
    def _set_default_params(self):
        """设置默认参数"""
        super()._set_default_params()
        
        # 覆盖SQL特定的模板
        sql_defaults = {
            "critique_template": "请评估以下SQL查询的质量，并指出语法错误、优化机会或其他需要改进的地方：\n\n```sql\n{answer}\n```\n\n评估：",
            "improve_template": "请根据以下评估改进SQL查询：\n\n原始查询：\n```sql\n{answer}\n```\n\n评估：{critique}\n\n改进后的查询（只需提供SQL代码）：",
            "max_iterations": 2  # SQL通常只需要较少的迭代
        }
        
        # 更新超参数
        for k, v in sql_defaults.items():
            self.hyperparameters[k] = v 