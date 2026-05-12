#!/usr/bin/env python3
"""DSCLR-Entropy v2 grid evaluation across Core17/Robust04/News21."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from eval.metrics.evaluator import DataLoader, FollowIREvaluator
from eval.models.encoder import ModelFactory

ROOT = Path('/home/luwa/Documents/DSCLR')
DUAL_DIR = ROOT / 'dataset' / 'FollowIR_test' / 'dual_queries_v5'
EMB_DIR = ROOT / 'dataset' / 'FollowIR_test' / 'embeddings' / 'RepLLaMA_reproduced'

TASKS = [
    'Core17InstructionRetrieval',
    'Robust04InstructionRetrieval',
    'News21InstructionRetrieval',
]


@dataclass
class QueryCache:
    qid: str
    doc_ids: List[str]
    s_pos: torch.Tensor
    s_neg: torch.Tensor
    mu: float
    sigma: float
    gap: float
    has_neg: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run DSCLR-Entropy eta x lambda grid evaluation.')
    parser.add_argument('--model_name', default='samaya-ai/RepLLaMA-reproduced')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lambda_penalty', type=float, default=0.05)
    parser.add_argument('--lambdas', default='')
    parser.add_argument('--etas', default='0.0,0.5,0.8,1.0,1.2,1.5,2.0,2.5,3.0')
    parser.add_argument('--target_p_mrr', type=float, default=0.15)
    parser.add_argument('--target_core_map', type=float, default=0.23)
    parser.add_argument('--target_robust_map', type=float, default=0.28)
    parser.add_argument('--target_news_ndcg', type=float, default=0.29)
    parser.add_argument('--output_dir', default=str(ROOT / 'evaluation' / 'dsclr' / 'entropy_grid' / datetime.now().strftime('%Y%m%d_%H%M%S')))
    return parser.parse_args()


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


def dual_file(task: str) -> Path:
    path = DUAL_DIR / f'dual_queries_v5_{task}.jsonl'
    if path.exists():
        return path
    fallback = ROOT / 'dataset' / 'FollowIR_test' / 'dual_queries' / f'dual_queries_{task}.jsonl'
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f'No dual query file for {task}')


def load_dual_changed(task: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(dual_file(task), 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            qid = row.get('qid', '')
            if qid.endswith('-changed'):
                out[qid] = row
    return out


def to_tensor(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def build_og_results(
    q_og: Dict[str, str],
    candidates: Dict[str, List[str]],
    encoder,
    corpus_emb: torch.Tensor,
    doc_to_idx: Dict[str, int],
    device: str,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    qids = sorted(q_og.keys(), key=lambda x: int(x.split('-')[0]))
    texts = [q_og[qid] for qid in qids]
    q_emb = to_tensor(encoder.encode_queries(texts, batch_size=batch_size), device)

    results: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        base = qid.split('-')[0]
        doc_ids = [d for d in candidates.get(base, [])[:1000] if d in doc_to_idx]
        if not doc_ids:
            continue
        idxs = [doc_to_idx[d] for d in doc_ids]
        scores = torch.mv(corpus_emb[idxs], q_emb[i]).detach().cpu().numpy()
        results[qid] = {d: float(s) for d, s in zip(doc_ids, scores)}
    return results


def build_changed_cache(
    q_changed: Dict[str, str],
    dual_changed: Dict[str, Dict[str, Any]],
    candidates: Dict[str, List[str]],
    encoder,
    corpus_emb: torch.Tensor,
    doc_to_idx: Dict[str, int],
    device: str,
    batch_size: int,
) -> List[QueryCache]:
    changed_qids = sorted(q_changed.keys(), key=lambda x: int(x.split('-')[0]))
    q_plus_texts: List[str] = []
    q_minus_texts: List[str] = []
    has_neg: List[bool] = []
    used_qids: List[str] = []

    for qid in changed_qids:
        row = dual_changed.get(qid, {})
        q_plus = str(row.get('q_plus', '')).strip() or q_changed[qid]
        q_minus = str(row.get('q_minus', '')).strip()
        used_qids.append(qid)
        q_plus_texts.append(q_plus)
        if q_minus in ('', '[NONE]'):
            q_minus_texts.append('none')
            has_neg.append(False)
        else:
            q_minus_texts.append(q_minus)
            has_neg.append(True)

    q_plus_emb = to_tensor(encoder.encode_queries(q_plus_texts, batch_size=batch_size), device)
    q_minus_emb = to_tensor(encoder.encode_queries(q_minus_texts, batch_size=batch_size), device)

    cache: List[QueryCache] = []
    for i, qid in enumerate(used_qids):
        base = qid.split('-')[0]
        doc_ids = [d for d in candidates.get(base, [])[:1000] if d in doc_to_idx]
        if not doc_ids:
            continue
        idxs = [doc_to_idx[d] for d in doc_ids]
        docs = corpus_emb[idxs]
        s_pos = torch.mv(docs, q_plus_emb[i])
        s_neg = torch.mv(docs, q_minus_emb[i])
        mu = float(s_neg.mean().item())
        sigma = float(s_neg.std(unbiased=False).item())
        gap = float(torch.max(s_pos).item() - torch.max(s_neg).item())
        cache.append(QueryCache(
            qid=qid,
            doc_ids=doc_ids,
            s_pos=s_pos,
            s_neg=s_neg,
            mu=mu,
            sigma=sigma,
            gap=gap,
            has_neg=has_neg[i],
        ))

    return cache


def dsclr_entropy_v2_rerank(cache: QueryCache, eta: float, lam: float) -> Dict[str, float]:
    if not cache.has_neg:
        s_final = cache.s_pos
    else:
        # 1) Query-wise z-score normalization
        z_neg = (cache.s_neg - cache.mu) / (cache.sigma + 1e-6)

        # 2) Gap-adaptive hammer with a hard cap for numerical stability
        alpha_adaptive = lam / (cache.gap + 1e-6)
        if not np.isfinite(alpha_adaptive):
            alpha_adaptive = 0.0
        alpha_adaptive = min(alpha_adaptive, 5.0)

        # 3) Exponential penalty (Entropy 2.0)
        penalty = alpha_adaptive * torch.exp(z_neg - eta)
        s_final = cache.s_pos - penalty
        s_final = torch.nan_to_num(s_final, nan=0.0, posinf=0.0, neginf=0.0)

    scores = s_final.detach().cpu().numpy()
    return {d: float(s) for d, s in zip(cache.doc_ids, scores)}


def dataset_primary_metric(task: str, metrics: Dict[str, Any]) -> float:
    if 'News21' in task:
        return float(metrics['changed']['ndcg_at_5'])
    return float(metrics['changed']['map_at_1000'])


def run_task(task: str, args: argparse.Namespace, encoder, etas: List[float], lambdas: List[float]) -> Dict[str, Any]:
    dl = DataLoader(task)
    q_og, q_changed = dl.load_queries()
    candidates = dl.load_candidates()
    dual_changed = load_dual_changed(task)

    corpus_emb, doc_ids = load_cached_embeddings(task)
    corpus_emb = corpus_emb.to(args.device)
    doc_to_idx = {d: i for i, d in enumerate(doc_ids)}

    results_og = build_og_results(
        q_og=q_og,
        candidates=candidates,
        encoder=encoder,
        corpus_emb=corpus_emb,
        doc_to_idx=doc_to_idx,
        device=args.device,
        batch_size=args.batch_size,
    )

    changed_cache = build_changed_cache(
        q_changed=q_changed,
        dual_changed=dual_changed,
        candidates=candidates,
        encoder=encoder,
        corpus_emb=corpus_emb,
        doc_to_idx=doc_to_idx,
        device=args.device,
        batch_size=args.batch_size,
    )

    evaluator = FollowIREvaluator(task)

    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for lam in lambdas:
        for eta in etas:
            results_changed: Dict[str, Dict[str, float]] = {}
            for qc in changed_cache:
                results_changed[qc.qid] = dsclr_entropy_v2_rerank(qc, eta=eta, lam=lam)

            metrics = evaluator.evaluate(results_og=results_og, results_changed=results_changed)
            row = {
                'task': task,
                'eta': eta,
                'lambda_penalty': lam,
                'p-MRR': float(metrics['p-MRR']),
                'changed_map_at_1000': float(metrics['changed']['map_at_1000']),
                'changed_ndcg_at_5': float(metrics['changed']['ndcg_at_5']),
                'og_map_at_1000': float(metrics['original']['map_at_1000']),
                'og_ndcg_at_5': float(metrics['original']['ndcg_at_5']),
                'primary_metric': dataset_primary_metric(task, metrics),
            }
            rows.append(row)

            if best is None or row['primary_metric'] > best['primary_metric']:
                best = row

    return {
        'task': task,
        'rows': rows,
        'best': best,
    }


def summarize_cross_task(payload: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}
    for task_block in payload['tasks']:
        task = task_block['task']
        for row in task_block['rows']:
            key = (float(row['eta']), float(row['lambda_penalty']))
            grouped.setdefault(key, {})[task] = row

    combo_rows: List[Dict[str, Any]] = []
    for (eta, lam), by_task in grouped.items():
        if any(task not in by_task for task in TASKS):
            continue

        core = by_task['Core17InstructionRetrieval']
        robust = by_task['Robust04InstructionRetrieval']
        news = by_task['News21InstructionRetrieval']

        min_p_mrr = min(core['p-MRR'], robust['p-MRR'], news['p-MRR'])
        mean_p_mrr = (core['p-MRR'] + robust['p-MRR'] + news['p-MRR']) / 3.0
        mean_map = (core['changed_map_at_1000'] + robust['changed_map_at_1000'] + news['changed_map_at_1000']) / 3.0
        mean_ndcg = (core['changed_ndcg_at_5'] + robust['changed_ndcg_at_5'] + news['changed_ndcg_at_5']) / 3.0

        pass_all_targets = (
            core['p-MRR'] >= args.target_p_mrr
            and robust['p-MRR'] >= args.target_p_mrr
            and news['p-MRR'] >= args.target_p_mrr
            and core['changed_map_at_1000'] >= args.target_core_map
            and robust['changed_map_at_1000'] >= args.target_robust_map
            and news['changed_ndcg_at_5'] >= args.target_news_ndcg
        )

        # Ratio score: >1 means hitting targets, min-ratio prioritizes the weakest metric.
        ratio_core = min(core['p-MRR'] / args.target_p_mrr, core['changed_map_at_1000'] / args.target_core_map)
        ratio_robust = min(robust['p-MRR'] / args.target_p_mrr, robust['changed_map_at_1000'] / args.target_robust_map)
        ratio_news = min(news['p-MRR'] / args.target_p_mrr, news['changed_ndcg_at_5'] / args.target_news_ndcg)
        min_target_ratio = min(ratio_core, ratio_robust, ratio_news)
        avg_target_ratio = (ratio_core + ratio_robust + ratio_news) / 3.0

        combo_rows.append({
            'eta': eta,
            'lambda_penalty': lam,
            'core_p-MRR': core['p-MRR'],
            'core_changed_map_at_1000': core['changed_map_at_1000'],
            'core_changed_ndcg_at_5': core['changed_ndcg_at_5'],
            'robust_p-MRR': robust['p-MRR'],
            'robust_changed_map_at_1000': robust['changed_map_at_1000'],
            'robust_changed_ndcg_at_5': robust['changed_ndcg_at_5'],
            'news_p-MRR': news['p-MRR'],
            'news_changed_map_at_1000': news['changed_map_at_1000'],
            'news_changed_ndcg_at_5': news['changed_ndcg_at_5'],
            'min_p-MRR': min_p_mrr,
            'mean_p-MRR': mean_p_mrr,
            'mean_changed_map_at_1000': mean_map,
            'mean_changed_ndcg_at_5': mean_ndcg,
            'pass_all_targets': int(pass_all_targets),
            'min_target_ratio': min_target_ratio,
            'avg_target_ratio': avg_target_ratio,
        })

    combo_rows.sort(key=lambda r: (r['pass_all_targets'], r['min_target_ratio'], r['avg_target_ratio']), reverse=True)
    return combo_rows


def save_outputs(output_dir: Path, payload: Dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'entropy_grid_results.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    all_rows: List[Dict[str, Any]] = []
    for t in payload['tasks']:
        all_rows.extend(t['rows'])

    fieldnames = [
        'task', 'eta', 'lambda_penalty', 'p-MRR',
        'changed_map_at_1000', 'changed_ndcg_at_5',
        'og_map_at_1000', 'og_ndcg_at_5', 'primary_metric'
    ]
    with open(output_dir / 'entropy_grid_all_rows.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    best_rows = [t['best'] for t in payload['tasks'] if t.get('best')]
    with open(output_dir / 'entropy_grid_best_by_task.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    cross_rows = summarize_cross_task(payload, args)
    if cross_rows:
        cross_fields = list(cross_rows[0].keys())
        with open(output_dir / 'entropy_grid_cross_task_summary.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cross_fields)
            writer.writeheader()
            writer.writerows(cross_rows)

        with open(output_dir / 'entropy_grid_best_unified.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_unified': cross_rows[0],
                'best_pass_all_targets': next((r for r in cross_rows if r['pass_all_targets'] == 1), None),
                'targets': {
                    'p_mrr': args.target_p_mrr,
                    'core_map': args.target_core_map,
                    'robust_map': args.target_robust_map,
                    'news_ndcg': args.target_news_ndcg,
                },
            }, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    etas = [float(x.strip()) for x in args.etas.split(',') if x.strip()]
    if args.lambdas.strip():
        lambdas = [float(x.strip()) for x in args.lambdas.split(',') if x.strip()]
    else:
        lambdas = [float(args.lambda_penalty)]

    encoder = ModelFactory.create(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )

    task_results = []
    for task in TASKS:
        print(f'Running DSCLR-Entropy grid for {task} ...')
        result = run_task(task=task, args=args, encoder=encoder, etas=etas, lambdas=lambdas)
        task_results.append(result)
        best = result['best']
        if best:
            print(
                f"  best eta={best['eta']:.3f}, lambda={best['lambda_penalty']:.4f}, p-MRR={best['p-MRR']:.4f}, "
                f"MAP={best['changed_map_at_1000']:.4f}, nDCG@5={best['changed_ndcg_at_5']:.4f}"
            )

    payload = {
        'model_name': args.model_name,
        'device': args.device,
        'lambdas': lambdas,
        'etas': etas,
        'tasks': task_results,
    }

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, payload, args)

    cross_rows = summarize_cross_task(payload, args)
    if cross_rows:
        best = cross_rows[0]
        print(
            f"Unified best: eta={best['eta']:.3f}, lambda={best['lambda_penalty']:.4f}, "
            f"pass_all_targets={best['pass_all_targets']}, min_target_ratio={best['min_target_ratio']:.4f}"
        )
    print(f'\nSaved outputs to: {output_dir}')


if __name__ == '__main__':
    main()
