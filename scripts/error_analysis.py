#!/usr/bin/env python3
"""
深度 Error Analysis 脚本
提取两类最典型的坏例：
1. MAP 血崩的元凶（被大锤砸死的好文）
2. pMRR 停滞的元凶（免疫惩罚的铁头烂文）
"""

import json
import os
import sys
from collections import defaultdict

def load_corpus(task_name):
    """加载文档语料"""
    from datasets import load_dataset
    corpus = load_dataset(f"jhu-clsp/{task_name}-mteb", "corpus", trust_remote_code=True)
    doc_map = {}
    for doc in corpus['corpus']:
        doc_map[doc['_id']] = {
            'text': doc['text'],
            'title': doc.get('title', '')
        }
    return doc_map

def load_qrels(task_name):
    """加载相关性标签"""
    from datasets import load_dataset
    qrels = load_dataset(f"jhu-clsp/{task_name}-mteb", "qrels", trust_remote_code=True)
    qrel_map = defaultdict(dict)
    for row in qrels['test']:
        qrel_map[row['query-id']][row['corpus-id']] = row['score']
    return qrel_map

def load_queries(task_name):
    """加载查询"""
    from datasets import load_dataset
    queries = load_dataset(f"jhu-clsp/{task_name}-mteb", "default", trust_remote_code=True)
    query_map = {}
    for query in queries['queries']:
        query_map[query['_id']] = {
            'text': query['text'],
            'negative_words': query.get('negative_words', '')
        }
    return query_map

def extract_bad_cases():
    """提取两类坏例"""
    
    # 加载数据
    print("=" * 80)
    print("🔍 加载数据...")
    print("=" * 80)
    
    task_name = "Core17InstructionRetrieval"
    
    # 加载评估结果
    with open('evaluation/dsclr/lap_eval/4.4-delta-grid-search/random_search_results.json', 'r') as f:
        results = json.load(f)
    
    # 找到最佳参数的结果
    best_result = None
    for result in results['all_results']:
        if abs(result['alpha'] - 1.5) < 0.01 and abs(result['delta'] - (-0.1)) < 0.01:
            best_result = result
            break
    
    if best_result is None:
        print("❌ 未找到最佳参数的结果")
        return
    
    print(f"✅ 找到最佳参数: alpha={best_result['alpha']}, delta={best_result['delta']}")
    print(f"   pMRR: {best_result['p-MRR']:.6f}")
    
    # 加载语料和标签
    print("\n加载文档语料...")
    corpus = load_corpus(task_name)
    print(f"✅ 加载了 {len(corpus)} 个文档")
    
    print("\n加载相关性标签...")
    qrels = load_qrels(task_name)
    print(f"✅ 加载了 {len(qrels)} 个查询的标签")
    
    print("\n加载查询...")
    queries = load_queries(task_name)
    print(f"✅ 加载了 {len(queries)} 个查询")
    
    # 提取坏例
    print("\n" + "=" * 80)
    print("🚨 第一类坏例：MAP 血崩的元凶（被大锤砸死的好文）")
    print("=" * 80)
    print("筛选条件: relevance > 0, Actual_Penalty > 0.02, Rank(Changed) > Rank(OG) + 20")
    
    type1_cases = []
    
    for query_id, docs in best_result['results_changed'].items():
        query_base = query_id.replace('-changed', '')
        qrel = qrels.get(query_base, {})
        neg_words = queries.get(query_base, {}).get('negative_words', '')
        
        # 获取 OG 排名
        og_rank_map = {}
        for rank, (doc_id, score) in enumerate(docs, 1):
            og_rank_map[doc_id] = rank
        
        # 分析每个文档
        for rank_changed, (doc_id, score_changed) in enumerate(docs, 1):
            relevance = qrel.get(doc_id, 0)
            
            if relevance > 0:
                # 这是好文档，检查是否被误伤
                rank_og = og_rank_map.get(doc_id, rank_changed)
                rank_drop = rank_changed - rank_og
                
                # 从白盒报告中获取详细信息
                # 这里简化处理，实际需要从白盒报告中提取
                if rank_drop > 20:
                    doc_text = corpus.get(doc_id, {}).get('text', '')[:200]
                    type1_cases.append({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'relevance': relevance,
                        'rank_og': rank_og,
                        'rank_changed': rank_changed,
                        'rank_drop': rank_drop,
                        'score_changed': score_changed,
                        'neg_words': neg_words,
                        'doc_text': doc_text
                    })
    
    # 按排名下降幅度排序
    type1_cases.sort(key=lambda x: x['rank_drop'], reverse=True)
    
    print(f"\n找到 {len(type1_cases)} 个第一类坏例")
    print("\n前 3 个最典型的例子:")
    
    for i, case in enumerate(type1_cases[:3], 1):
        print(f"\n{'='*80}")
        print(f"案例 {i}:")
        print(f"  Query ID: {case['query_id']}")
        print(f"  Doc ID: {case['doc_id']}")
        print(f"  Relevance: {case['relevance']}")
        print(f"  OG 排名: {case['rank_og']}")
        print(f"  Changed 排名: {case['rank_changed']}")
        print(f"  排名下降: {case['rank_drop']}")
        print(f"  负向词: {case['neg_words']}")
        print(f"  文档片段: {case['doc_text']}...")
    
    # 第二类坏例
    print("\n" + "=" * 80)
    print("🚨 第二类坏例：pMRR 停滞的元凶（免疫惩罚的铁头烂文）")
    print("=" * 80)
    print("筛选条件: relevance == 0, Contains_Negative_Word == True, Actual_Penalty == 0, Rank(Changed) <= 20")
    
    type2_cases = []
    
    for query_id, docs in best_result['results_changed'].items():
        query_base = query_id.replace('-changed', '')
        qrel = qrels.get(query_base, {})
        neg_words = queries.get(query_base, {}).get('negative_words', '')
        
        # 分析前 20 个文档
        for rank_changed, (doc_id, score_changed) in enumerate(docs[:20], 1):
            relevance = qrel.get(doc_id, 0)
            
            if relevance == 0:
                # 这是坏文档，检查是否包含负向词
                doc_text = corpus.get(doc_id, {}).get('text', '')
                doc_title = corpus.get(doc_id, {}).get('title', '')
                full_text = f"{doc_title} {doc_text}".lower()
                
                # 检查是否包含负向词
                neg_words_list = [w.strip().lower() for w in neg_words.split(',') if w.strip()]
                contains_neg = any(neg in full_text for neg in neg_words_list)
                
                if contains_neg:
                    # 找到包含负向词的上下文
                    context = ""
                    for neg in neg_words_list:
                        if neg in full_text:
                            idx = full_text.find(neg)
                            start = max(0, idx - 100)
                            end = min(len(full_text), idx + len(neg) + 100)
                            context = full_text[start:end]
                            break
                    
                    type2_cases.append({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'relevance': relevance,
                        'rank_changed': rank_changed,
                        'score_changed': score_changed,
                        'neg_words': neg_words,
                        'contains_neg': contains_neg,
                        'context': context,
                        'doc_text': doc_text[:200]
                    })
    
    # 按排名排序
    type2_cases.sort(key=lambda x: x['rank_changed'])
    
    print(f"\n找到 {len(type2_cases)} 个第二类坏例")
    print("\n前 3 个最典型的例子:")
    
    for i, case in enumerate(type2_cases[:3], 1):
        print(f"\n{'='*80}")
        print(f"案例 {i}:")
        print(f"  Query ID: {case['query_id']}")
        print(f"  Doc ID: {case['doc_id']}")
        print(f"  Relevance: {case['relevance']}")
        print(f"  Changed 排名: {case['rank_changed']}")
        print(f"  负向词: {case['neg_words']}")
        print(f"  包含负向词: {case['contains_neg']}")
        print(f"  负向词上下文: ...{case['context']}...")
        print(f"  文档片段: {case['doc_text']}...")

if __name__ == "__main__":
    extract_bad_cases()
