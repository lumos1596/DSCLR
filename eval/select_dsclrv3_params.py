"""
DSCLR-V3 参数选择器

用途：
- 从 all_results.json 中按业务目标选择“可上线参数”
- 支持硬约束过滤（如 p-MRR 下限、最大允许降幅）
- 输出结构化 JSON 与可读 Markdown 报告
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def _metric_key(objective: str) -> str:
    mapping = {
        "changed_map": "changed_MAP@1000",
        "changed_ndcg5": "changed_nDCG@5",
        "changed_ndcg10": "changed_nDCG@10",
        "pmrr": "p-MRR",
    }
    if objective not in mapping:
        raise ValueError(f"不支持的 objective: {objective}")
    return mapping[objective]


def _safe_get(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        return float(val)
    except Exception:
        return default


def _build_composite(row: Dict[str, Any]) -> float:
    """默认综合目标：更偏重 changed MAP，同时兼顾 p-MRR 与 nDCG@5。"""
    changed_map = _safe_get(row, "changed_MAP@1000")
    pmrr = _safe_get(row, "p-MRR")
    changed_ndcg5 = _safe_get(row, "changed_nDCG@5")
    return 0.6 * changed_map + 0.25 * pmrr + 0.15 * changed_ndcg5


def select_candidates(
    all_results: List[Dict[str, Any]],
    objective: str,
    min_pmrr: Optional[float] = None,
    min_changed_map: Optional[float] = None,
    min_changed_ndcg5: Optional[float] = None,
    baseline_pmrr: Optional[float] = None,
    max_pmrr_drop: Optional[float] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    total = len(all_results)

    feasible: List[Dict[str, Any]] = []
    for row in all_results:
        pmrr = _safe_get(row, "p-MRR")
        changed_map = _safe_get(row, "changed_MAP@1000")
        changed_ndcg5 = _safe_get(row, "changed_nDCG@5")

        if min_pmrr is not None and pmrr < min_pmrr:
            continue
        if min_changed_map is not None and changed_map < min_changed_map:
            continue
        if min_changed_ndcg5 is not None and changed_ndcg5 < min_changed_ndcg5:
            continue

        if baseline_pmrr is not None and max_pmrr_drop is not None:
            pmrr_drop = baseline_pmrr - pmrr
            if pmrr_drop > max_pmrr_drop:
                continue

        candidate = dict(row)
        candidate["objective_score"] = (
            _build_composite(candidate)
            if objective == "composite"
            else _safe_get(candidate, _metric_key(objective))
        )
        if baseline_pmrr is not None:
            candidate["pmrr_drop_vs_baseline"] = baseline_pmrr - pmrr
        feasible.append(candidate)

    feasible_sorted = sorted(
        feasible,
        key=lambda x: (
            _safe_get(x, "objective_score"),
            _safe_get(x, "changed_MAP@1000"),
            _safe_get(x, "p-MRR"),
        ),
        reverse=True,
    )

    best = feasible_sorted[0] if feasible_sorted else None
    top = feasible_sorted[: max(top_k, 1)]

    return {
        "total_trials": total,
        "feasible_trials": len(feasible_sorted),
        "objective": objective,
        "constraints": {
            "min_pmrr": min_pmrr,
            "min_changed_map": min_changed_map,
            "min_changed_ndcg5": min_changed_ndcg5,
            "baseline_pmrr": baseline_pmrr,
            "max_pmrr_drop": max_pmrr_drop,
        },
        "best": best,
        "top_candidates": top,
    }


def _format_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DSCLR-V3 目标导向选参报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().isoformat()}")
    lines.append(f"- 总试验数: {report['total_trials']}")
    lines.append(f"- 满足约束数: {report['feasible_trials']}")
    lines.append(f"- 目标函数: {report['objective']}")
    lines.append("")

    constraints = report["constraints"]
    lines.append("## 约束")
    for k, v in constraints.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    if report["best"] is None:
        lines.append("## 结果")
        lines.append("- 没有满足约束的参数组合。")
        return "\n".join(lines)

    best = report["best"]
    lines.append("## 最佳参数")
    lines.append(f"- beta_min: {best.get('beta_min')}")
    lines.append(f"- gamma: {best.get('gamma')}")
    lines.append(f"- alpha: {best.get('alpha')}")
    lines.append(f"- theta: {best.get('theta')}")
    lines.append(f"- paradox_threshold: {best.get('paradox_threshold')}")
    lines.append("")
    lines.append("## 最佳指标")
    lines.append(f"- objective_score: {best.get('objective_score'):.8f}")
    lines.append(f"- p-MRR: {_safe_get(best, 'p-MRR'):.8f}")
    lines.append(f"- changed_MAP@1000: {_safe_get(best, 'changed_MAP@1000'):.8f}")
    lines.append(f"- changed_nDCG@5: {_safe_get(best, 'changed_nDCG@5'):.8f}")
    lines.append(f"- changed_nDCG@10: {_safe_get(best, 'changed_nDCG@10'):.8f}")
    if "pmrr_drop_vs_baseline" in best:
        lines.append(f"- pmrr_drop_vs_baseline: {_safe_get(best, 'pmrr_drop_vs_baseline'):.8f}")
    lines.append("")

    lines.append("## Top 候选")
    for i, row in enumerate(report["top_candidates"], start=1):
        parts = [
            f"{i}. beta_min={row.get('beta_min')}",
            f"gamma={row.get('gamma')}",
            f"alpha={row.get('alpha')}",
            f"theta={row.get('theta')}",
            f"paradox_threshold={row.get('paradox_threshold')}",
            f"objective_score={_safe_get(row, 'objective_score'):.8f}",
            f"p-MRR={_safe_get(row, 'p-MRR'):.8f}",
            f"changed_MAP@1000={_safe_get(row, 'changed_MAP@1000'):.8f}",
        ]
        lines.append("- " + ", ".join(parts))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSCLR-V3 参数筛选器")
    parser.add_argument("--all_results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--objective",
        type=str,
        default="changed_map",
        choices=["changed_map", "changed_ndcg5", "changed_ndcg10", "pmrr", "composite"],
    )
    parser.add_argument("--min_pmrr", type=float, default=None)
    parser.add_argument("--min_changed_map", type=float, default=None)
    parser.add_argument("--min_changed_ndcg5", type=float, default=None)

    parser.add_argument("--baseline_pmrr", type=float, default=None)
    parser.add_argument("--max_pmrr_drop", type=float, default=None)

    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    with open(args.all_results, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    report = select_candidates(
        all_results=all_results,
        objective=args.objective,
        min_pmrr=args.min_pmrr,
        min_changed_map=args.min_changed_map,
        min_changed_ndcg5=args.min_changed_ndcg5,
        baseline_pmrr=args.baseline_pmrr,
        max_pmrr_drop=args.max_pmrr_drop,
        top_k=args.top_k,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "selected_params_report.json")
    md_path = os.path.join(args.output_dir, "selected_params_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_format_md(report))

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Feasible/Total: {report['feasible_trials']}/{report['total_trials']}")
    if report["best"] is not None:
        print(f"Best objective_score: {report['best']['objective_score']:.8f}")
    else:
        print("No feasible candidates under current constraints.")


if __name__ == "__main__":
    main()
