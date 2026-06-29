"""
调用 DeepSeek API 为带负向约束的 changed-query 批量生成 safe anchor 文档。

用法:
    python -m utils.call_llm.generate_safe_anchors \
        --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Robust04InstructionRetrieval.jsonl \
        --output_path dataset/FollowIR_test/safe_anchors/safe_anchors_robust04.json \
        --api_key sk-xxxx \
        --num_anchors 3

特性:
    - 自动筛选带负向约束的 changed-query
    - 断点续传：已生成的 query 自动跳过
    - 失败重试
    - JSON schema 输出，结构稳定
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 确保能 import 同目录的 call_deepseek
sys.path.insert(0, str(Path(__file__).resolve().parent))
from call_deepseek import call_deepseek  # noqa: E402


SYSTEM_PROMPT = """你是一位信息检索领域的专家助手。你的任务是为给定的检索查询生成"安全锚点文档"(safe anchor documents)。

安全锚点文档的定义: 符合原始查询主题(q_base)和正向约束(q_pos), 但不满足负向约束(q_neg)的"无辜文档"摘要。这些锚点用于估计负向惩罚的阈值, 不用于检索召回。

生成规则(严格遵守):
1. 锚点必须是文档摘要形式(1-2句话), 而非查询改写。以 "A report on..." / "An article describing..." / "A study of..." 等开头。
2. 每条锚点必须符合 q_base 主题, 并尽量体现 q_pos 中的正向要求(包括"必须提及X"类约束)。
3. 锚点不能真正满足 q_neg 的排除条件。但如果 q_base 与 q_neg 语义重叠(真实文档常会提及相关上下文), 锚点可以合理提及 q_neg 相关内容, 但必须明确说明不满足排除条件。例如: "虽然提及1990年的背景, 但重点分析2001年后的加固效果" / "文中虽出现X一词, 但仅作为地理标注, 不讨论X本身"。
4. 锚点应贴近真实文档分布, 包含具体的实体、时间、数据、机构名等细节, 使其看起来像真实存在的文档, 而非泛泛的描述。
5. 生成的 {num_anchors} 条锚点应有适度多样性, 覆盖不同的角度/子主题/时间段, 但都要满足上述规则。
6. 锚点用英文撰写(与文档语种一致)。

输出格式: 严格的 JSON, 形如:
{{"anchors": ["锚点1", "锚点2", "锚点3"]}}"""


def build_user_prompt(query: str, instruction: str, q_plus: str, q_minus: str,
                      num_anchors: int) -> str:
    return f"""查询信息:
- 原始查询(query): {query}
- 完整指令(instruction): {instruction}
- 正向约束(q_plus): {q_plus}
- 负向约束(q_minus): {q_minus}

请生成 {num_anchors} 条安全锚点文档。"""


def generate_anchors_for_query(api_key: str, query: str, instruction: str,
                               q_plus: str, q_minus: str, num_anchors: int,
                               max_retries: int = 3) -> Optional[List[str]]:
    """为单个 query 调用 API 生成锚点, 带重试。"""
    sys_prompt = SYSTEM_PROMPT.replace("{num_anchors}", str(num_anchors))
    user_prompt = build_user_prompt(query, instruction, q_plus, q_minus, num_anchors)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_deepseek(
                api_key=api_key,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                is_json=True,
                temperature=0.5,  # 略高温度增加多样性
            )
            if not raw:
                print(f"    [attempt {attempt}] 返回空, 重试...")
                time.sleep(2 * attempt)
                continue
            data = json.loads(raw)
            anchors = data.get("anchors", [])
            if not isinstance(anchors, list) or len(anchors) < num_anchors:
                print(f"    [attempt {attempt}] 锚点数不足: {len(anchors) if isinstance(anchors, list) else 'N/A'}, 重试...")
                time.sleep(2 * attempt)
                continue
            # 截断到所需数量, 去除空白
            anchors = [a.strip() for a in anchors[:num_anchors] if a and a.strip()]
            if len(anchors) < num_anchors:
                print(f"    [attempt {attempt}] 有效锚点不足, 重试...")
                time.sleep(2 * attempt)
                continue
            return anchors
        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}] JSON 解析失败: {e}, 重试...")
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"    [attempt {attempt}] 错误: {e}, 重试...")
            time.sleep(2 * attempt)
    return None


def load_existing(output_path: str) -> Dict[str, List[str]]:
    """加载已有输出用于断点续传。"""
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 去掉 _meta
    return {k: v for k, v in data.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description="用 DeepSeek API 生成 safe anchor 文档")
    parser.add_argument("--dual_queries_path", type=str, required=True,
                        help="dual_queries_v6 jsonl 路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出 JSON 路径")
    parser.add_argument("--api_key", type=str, default=None,
                        help="DeepSeek API key (默认读 DEEPSEEK_API_KEY 环境变量)")
    parser.add_argument("--num_anchors", type=int, default=3,
                        help="每个 query 生成锚点数 (默认 3)")
    parser.add_argument("--sleep_sec", type=float, default=0.5,
                        help="每次 API 调用间隔秒数 (默认 0.5)")
    parser.add_argument("--task_label", type=str, default=None,
                        help="任务标签, 用于日志显示")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未提供 API key, 请用 --api_key 或设置 DEEPSEEK_API_KEY 环境变量")

    # 读取 dual_queries
    records = []
    with open(args.dual_queries_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qm = d.get("q_minus", "")
            if (d.get("qid", "").endswith("-changed")
                    and qm and qm != "[NONE]" and qm.strip()):
                records.append(d)

    label = args.task_label or Path(args.dual_queries_path).stem
    print(f"[{label}] 带负向约束的 changed-query 共 {len(records)} 条")

    # 断点续传
    existing = load_existing(args.output_path)
    if existing:
        print(f"[{label}] 已有锚点 {len(existing)} 条, 将跳过")

    results = dict(existing)

    success = 0
    fail = 0
    for i, d in enumerate(records, 1):
        qid = d["qid"]
        if qid in results and len(results[qid]) >= args.num_anchors:
            continue
        print(f"[{i}/{len(records)}] 生成 {qid} ...")
        anchors = generate_anchors_for_query(
            api_key=api_key,
            query=d.get("query", ""),
            instruction=d.get("instruction", ""),
            q_plus=d.get("q_plus", ""),
            q_minus=d.get("q_minus", ""),
            num_anchors=args.num_anchors,
        )
        if anchors:
            results[qid] = anchors
            success += 1
            for a in anchors:
                print(f"    - {a[:100]}{'...' if len(a) > 100 else ''}")
        else:
            fail += 1
            print(f"    !! {qid} 生成失败")

        # 每生成一条就保存, 防止中断丢失
        _save(args.output_path, results, label)
        time.sleep(args.sleep_sec)

    _save(args.output_path, results, label)
    print(f"\n[{label}] 完成: 成功 {success}, 失败 {fail}, 总计 {len(results)} 条")


def _save(output_path: str, results: Dict[str, List[str]], label: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = {
        "_meta": {
            "description": f"DeepSeek-API-generated safe anchor documents for {label} changed-queries with negative constraints. Each anchor is an innocent/non-violating document summary that matches q_base and q_pos but does not satisfy q_neg. Used to calibrate the safe_anchor_threshold tau.",
            "num_anchors_per_query": 3,
            "generator": "deepseek-chat",
        }
    }
    out.update(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
