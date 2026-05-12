#!/usr/bin/env python3
"""Compute space fingerprint distributions for Core17 / Robust04 / News21.

Outputs a JSON and Markdown report under analysis_output/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from eval.metrics.evaluator import DataLoader
from eval.models.encoder import ModelFactory

ROOT = Path('/home/luwa/Documents/DSCLR')
DUAL_DIR = ROOT / 'dataset' / 'FollowIR_test' / 'dual_queries_v5'
EMB_DIR = ROOT / 'dataset' / 'FollowIR_test' / 'embeddings' / 'RepLLaMA_reproduced'
OUT_JSON = ROOT / 'analysis_output' / 'space_fingerprint_report.json'
OUT_MD = ROOT / 'analysis_output' / 'space_fingerprint_report.md'
TASKS = [
    'Core17InstructionRetrieval',
    'Robust04InstructionRetrieval',
    'News21InstructionRetrieval',
]


def dual_file(task: str) -> Path:
    path = DUAL_DIR / f'dual_queries_v5_{task}.jsonl'
    if path.exists():
        return path
    fallback = ROOT / 'dataset' / 'FollowIR_test' / 'dual_queries' / f'dual_queries_{task}.jsonl'
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f'No dual query file for {task}')


def load_dual_queries(task: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(dual_file(task), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get('qid', '').endswith('-changed')]
    rows.sort(key=lambda r: int(r['qid'].split('-')[0]))
    return rows


def load_cached_embeddings(task: str) -> Tuple[torch.Tensor, List[str]]:
    emb_path = EMB_DIR / f'{task}_RepLLaMA_reproduced_corpus_embeddings.npy'
    ids_path = EMB_DIR / f'{task}_RepLLaMA_reproduced_corpus_ids.json'
    emb = np.load(emb_path, allow_pickle=True)
    with open(ids_path, 'r', encoding='utf-8') as f:
        doc_ids = json.load(f)
    if emb.ndim == 0:
        data = emb.item()
        emb = np.array([data[did] for did in doc_ids], dtype=np.float32)
    return torch.tensor(emb, dtype=torch.float32), doc_ids


def contains_neg(doc_text: str, neg_words: List[str]) -> bool:
    text = (doc_text or '').lower()
    return any((w and w != '[NONE]' and w.lower() in text) for w in neg_words)


def summarize(arr: List[float]) -> Dict[str, Any]:
    if not arr:
        return {k: None for k in ['count', 'mean', 'std', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'max']}
    a = np.asarray(arr, dtype=np.float64)
    return {
        'count': int(a.size),
        'mean': float(a.mean()),
        'std': float(a.std(ddof=0)),
        'min': float(a.min()),
        'p10': float(np.percentile(a, 10)),
        'p25': float(np.percentile(a, 25)),
        'median': float(np.percentile(a, 50)),
        'p75': float(np.percentile(a, 75)),
        'p90': float(np.percentile(a, 90)),
        'p95': float(np.percentile(a, 95)),
        'max': float(a.max()),
    }


def feasible_threshold(good_z: List[float], bad_z: List[float]) -> Dict[str, Any]:
    if not good_z or not bad_z:
        return {'feasible': False}
    good_p95 = float(np.percentile(good_z, 95))
    bad_p10 = float(np.percentile(bad_z, 10))
    feasible = good_p95 <= bad_p10
    k = float((good_p95 + bad_p10) / 2.0) if feasible else None
    return {
        'feasible': feasible,
        'good_p95': good_p95,
        'bad_p10': bad_p10,
        'k_suggested': k,
        'bad_recall_at_k_suggested': float(np.mean(np.asarray(bad_z) > k)) if feasible else None,
        'good_fpr_at_k_suggested': float(np.mean(np.asarray(good_z) > k)) if feasible else None,
    }


def analyze_task(task: str, encoder, device: str) -> Dict[str, Any]:
    dl = DataLoader(task)
    qrels = dl.load_qrels()
    qrel_diff = dl.load_qrel_diff()
    corpus = dl.load_corpus()
    candidates = dl.load_candidates()
    dual_rows = load_dual_queries(task)
    qid_to_row = {r['qid']: r for r in dual_rows}

    corpus_emb, doc_ids = load_cached_embeddings(task)
    corpus_emb = corpus_emb.to(device)
    doc_to_idx = {did: i for i, did in enumerate(doc_ids)}

    qids: List[str] = []
    q_plus_texts: List[str] = []
    q_minus_texts: List[str] = []
    for row in dual_rows:
        qid = row['qid']
        q_minus = str(row.get('q_minus', '')).strip()
        if q_minus in ('', '[NONE]'):
            continue
        if qid not in qrels:
            continue
        base = qid.split('-')[0]
        if base not in candidates:
            continue
        q_plus = str(row.get('q_plus', '')).strip()
        if not q_plus:
            continue
        qids.append(qid)
        q_plus_texts.append(q_plus)
        q_minus_texts.append(q_minus)

    q_plus_embs = torch.tensor(encoder.encode_queries(q_plus_texts, batch_size=64), dtype=torch.float32, device=device)
    q_minus_embs = torch.tensor(encoder.encode_queries(q_minus_texts, batch_size=64), dtype=torch.float32, device=device)

    query_mu: List[float] = []
    query_sigma: List[float] = []
    gaps: List[float] = []
    good_z: List[float] = []
    bad_z: List[float] = []
    bad_top_percentiles: List[float] = []
    per_query: List[Dict[str, Any]] = []

    for idx, qid in enumerate(qids):
        base = qid.split('-')[0]
        cand_ids = [d for d in candidates.get(base, [])[:1000] if d in doc_to_idx]
        if not cand_ids:
            continue

        cand_idx = [doc_to_idx[d] for d in cand_ids]
        doc_mat = corpus_emb[cand_idx]
        qp = q_plus_embs[idx]
        qm = q_minus_embs[idx]

        s_pos = torch.mv(doc_mat, qp).detach().cpu().numpy().astype(np.float64)
        s_neg = torch.mv(doc_mat, qm).detach().cpu().numpy().astype(np.float64)
        mu = float(s_neg.mean())
        sigma = float(s_neg.std(ddof=0))
        z = (s_neg - mu) / (sigma + 1e-12)

        rels = qrels.get(qid, {})
        neg_words = [w.strip() for w in str(qid_to_row[qid].get('q_minus', '')).split(',') if w.strip() and w.strip() != '[NONE]']
        rel_mask = np.array([rels.get(d, 0) > 0 for d in cand_ids], dtype=bool)
        bad_mask = np.array([rels.get(d, 0) == 0 and contains_neg(corpus.get(d, {}).get('text', ''), neg_words) for d in cand_ids], dtype=bool)

        rel_z = z[rel_mask]
        b_z = z[bad_mask]
        good_z.extend(rel_z.tolist())
        bad_z.extend(b_z.tolist())

        order = np.argsort(-s_neg)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(s_neg) + 1)
        b_pct = [100.0 * ranks[j] / len(s_neg) for j in np.where(bad_mask)[0]]
        bad_top_percentiles.extend(b_pct)

        gap = float(np.max(s_pos) - np.max(s_neg))
        query_mu.append(mu)
        query_sigma.append(sigma)
        gaps.append(gap)

        per_query.append({
            'qid': qid,
            'query_mu': mu,
            'query_sigma': sigma,
            'gap': gap,
            'candidate_count': int(len(cand_ids)),
            'relevant_count': int(rel_mask.sum()),
            'bad_count': int(bad_mask.sum()),
            'relevant_z_mean': float(rel_z.mean()) if rel_z.size else None,
            'relevant_z_p95': float(np.percentile(rel_z, 95)) if rel_z.size else None,
            'bad_z_mean': float(b_z.mean()) if b_z.size else None,
            'bad_z_p95': float(np.percentile(b_z, 95)) if b_z.size else None,
            'bad_top_percentile_median': float(np.median(b_pct)) if b_pct else None,
            'bad_top_percentile_p90': float(np.percentile(b_pct, 90)) if b_pct else None,
        })

    task_summary = {
        'queries_loaded': len(dual_rows),
        'queries_analyzed': len(per_query),
        'query_mu_mean': float(np.mean(query_mu)) if query_mu else None,
        'query_sigma_mean': float(np.mean(query_sigma)) if query_sigma else None,
        'gap': summarize(gaps),
        'relevant_z': summarize(good_z),
        'bad_z': summarize(bad_z),
        'bad_top_percentile': summarize(bad_top_percentiles),
        'relevant_z_le_1_frac': float(np.mean(np.asarray(good_z) <= 1.0)) if good_z else None,
        'bad_z_gt_25_frac': float(np.mean(np.asarray(bad_z) > 2.5)) if bad_z else None,
        'threshold_window': feasible_threshold(good_z, bad_z),
    }

    return {
        'task': task,
        'summary': task_summary,
        'per_query': per_query,
    }


def main() -> None:
    os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder = ModelFactory.create(
        model_name='samaya-ai/RepLLaMA-reproduced',
        device=device,
        batch_size=64,
        normalize_embeddings=True,
    )

    report: Dict[str, Any] = {'tasks': {}, 'cross_task_threshold': None}
    global_good_z: List[float] = []
    global_bad_z: List[float] = []

    for task in TASKS:
        res = analyze_task(task, encoder, device)
        report['tasks'][task] = res
        global_good_z.extend([x for x in (q['relevant_z_mean'] for q in res['per_query']) if x is not None])
        global_bad_z.extend([x for x in (q['bad_z_mean'] for q in res['per_query']) if x is not None])

    # Cross-task feasible window from exact pooled doc-level z distributions
    # Recompute pooled doc-level arrays from per-query z means is not exact, so we use per-task exact windows and intersect them.
    task_windows = []
    for task in TASKS:
        tw = report['tasks'][task]['summary']['threshold_window']
        if tw.get('feasible'):
            task_windows.append((tw['good_p95'], tw['bad_p10']))

    if task_windows:
        lower = max(w[0] for w in task_windows)
        upper = min(w[1] for w in task_windows)
        report['cross_task_threshold'] = {
            'feasible': lower <= upper,
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'k_suggested': float((lower + upper) / 2.0) if lower <= upper else None,
        }
    else:
        report['cross_task_threshold'] = {'feasible': False}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md: List[str] = []
    md.append('# 空间指纹分布报告')
    md.append('')
    md.append(f'- 模型: RepLLaMA-reproduced')
    md.append(f'- 候选池: Top 1000')
    md.append('')
    for task in TASKS:
        s = report['tasks'][task]['summary']
        md.append(f'## {task}')
        md.append(f'- 分析查询数: {s["queries_analyzed"]}')
        md.append(f'- 查询级 S_neg 基准: μ均值={s["query_mu_mean"]:.4f}, σ均值={s["query_sigma_mean"]:.4f}')
        md.append(f'- Gap 均值={s["gap"]["mean"]:.4f}, 中位数={s["gap"]["median"]:.4f}, p25={s["gap"]["p25"]:.4f}, p75={s["gap"]["p75"]:.4f}')
        md.append(f'- 相关文档 Z: mean={s["relevant_z"]["mean"]:.4f}, p95={s["relevant_z"]["p95"]:.4f}, Z≤1 比例={s["relevant_z_le_1_frac"]:.4f}')
        md.append(f'- 烂文 Z: mean={s["bad_z"]["mean"]:.4f}, p50={s["bad_z"]["median"]:.4f}, p95={s["bad_z"]["p95"]:.4f}, Z>2.5 比例={s["bad_z_gt_25_frac"]:.4f}')
        md.append(f'- 烂文 top-percentile: median={s["bad_top_percentile"]["median"]:.2f}%, p90={s["bad_top_percentile"]["p90"]:.2f}%, 前3%比例={float(np.mean(np.asarray([q["bad_top_percentile_median"] for q in report["tasks"][task]["per_query"] if q["bad_top_percentile_median"] is not None]) <= 3.0)) if report["tasks"][task]["per_query"] else 0.0:.4f}')
        tw = s['threshold_window']
        if tw.get('feasible'):
            md.append(f'- 可行阈值窗: [{tw["good_p95"]:.4f}, {tw["bad_p10"]:.4f}]，建议 k≈{tw["k_suggested"]:.4f}')
        else:
            md.append('- 可行阈值窗: 不可行或样本不足')
        md.append('')

    md.append('## 三数据集交集阈值')
    ct = report['cross_task_threshold']
    if ct.get('feasible'):
        md.append(f'- 可行交集窗: [{ct["lower_bound"]:.4f}, {ct["upper_bound"]:.4f}]')
        md.append(f'- 建议统一 k: {ct["k_suggested"]:.4f}')
    else:
        md.append('- 三数据集同时满足“坏文召回≥90% 且好文误伤≤5%”的交集窗不存在')
    md.append('')
    md.append('## 说明')
    md.append('- Z-score 按查询内部 Top 1000 候选池计算。')
    md.append('- 烂文定义为 label=0 且命中该查询的 Q- 负向词。')
    md.append('- p95 / p10 阈值仅用于寻找统一 k 的可行区间。')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'SAVED {OUT_JSON}')
    print(f'SAVED {OUT_MD}')
    print(json.dumps(report['cross_task_threshold'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
