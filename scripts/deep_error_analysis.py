#!/usr/bin/env python3
"""
深度 Error Analysis 脚本
从白盒报告和评估结果中提取两类最典型的坏例
"""

import json
import re
from pathlib import Path

def parse_whitebox_report(report_path):
    """解析白盒报告"""
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    queries = []
    current_query = None
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('## 📋 Query'):
            if current_query:
                queries.append(current_query)
            
            query_id = line.split('Query ')[1].strip()
            current_query = {
                'query_id': query_id,
                'neg_words': '',
                'dynamic_tau': 0.0,
                'og_docs': [],
                'changed_docs': [],
                'penalized_docs': [],
                'escaped_docs': []
            }
        
        elif current_query:
            if line.startswith('- **负向词**'):
                current_query['neg_words'] = line.split(': ')[1].strip()
            elif line.startswith('- **Dynamic_Tau'):
                tau_str = line.split(': ')[1].strip()
                current_query['dynamic_tau'] = float(tau_str)
            
            # 解析 OG 模式文档
            elif line.startswith('### 🎯 OG 模式 - 相关文档排名'):
                i += 3  # 跳过表头
                while i < len(lines) and lines[i].strip().startswith('|'):
                    parts = lines[i].strip().split('|')
                    if len(parts) >= 5:
                        try:
                            rank = int(parts[1].strip())
                            doc_id = parts[2].strip()
                            relevance = int(parts[3].strip())
                            s_pos = float(parts[4].strip())
                            current_query['og_docs'].append({
                                'rank': rank,
                                'doc_id': doc_id,
                                'relevance': relevance,
                                's_pos': s_pos
                            })
                        except:
                            pass
                    i += 1
            
            # 解析 Changed 模式文档
            elif line.startswith('### 🎯 Changed 模式 - 相关文档排名'):
                i += 3  # 跳过表头
                while i < len(lines) and lines[i].strip().startswith('|'):
                    parts = lines[i].strip().split('|')
                    if len(parts) >= 5:
                        try:
                            rank = int(parts[1].strip())
                            doc_id = parts[2].strip()
                            relevance = int(parts[3].strip())
                            s_final = float(parts[4].strip())
                            current_query['changed_docs'].append({
                                'rank': rank,
                                'doc_id': doc_id,
                                'relevance': relevance,
                                's_final': s_final
                            })
                        except:
                            pass
                    i += 1
            
            # 解析被惩罚的文档
            elif line.startswith('### 📉 被惩罚最重的文档'):
                i += 3  # 跳过表头
                while i < len(lines) and lines[i].strip().startswith('|'):
                    parts = lines[i].strip().split('|')
                    if len(parts) >= 9:
                        try:
                            rank = int(parts[1].strip())
                            doc_id = parts[2].strip()
                            relevance = int(parts[3].strip())
                            s_pos = float(parts[4].strip())
                            s_neg_proj = float(parts[5].strip())
                            dynamic_tau = float(parts[6].strip())
                            over_threshold = float(parts[7].strip())
                            actual_penalty = float(parts[8].strip())
                            s_final = float(parts[9].strip())
                            current_query['penalized_docs'].append({
                                'rank': rank,
                                'doc_id': doc_id,
                                'relevance': relevance,
                                's_pos': s_pos,
                                's_neg_proj': s_neg_proj,
                                'dynamic_tau': dynamic_tau,
                                'over_threshold': over_threshold,
                                'actual_penalty': actual_penalty,
                                's_final': s_final
                            })
                        except:
                            pass
                    i += 1
            
            # 解析漏网烂文
            elif line.startswith('### ⚠️ 漏网烂文'):
                i += 3  # 跳过表头
                while i < len(lines) and lines[i].strip().startswith('|'):
                    parts = lines[i].strip().split('|')
                    if len(parts) >= 9:
                        try:
                            rank = int(parts[1].strip())
                            doc_id = parts[2].strip()
                            relevance = int(parts[3].strip())
                            s_pos = float(parts[4].strip())
                            s_neg_proj = float(parts[5].strip())
                            dynamic_tau = float(parts[6].strip())
                            over_threshold = float(parts[7].strip())
                            actual_penalty = float(parts[8].strip())
                            s_final = float(parts[9].strip())
                            current_query['escaped_docs'].append({
                                'rank': rank,
                                'doc_id': doc_id,
                                'relevance': relevance,
                                's_pos': s_pos,
                                's_neg_proj': s_neg_proj,
                                'dynamic_tau': dynamic_tau,
                                'over_threshold': over_threshold,
                                'actual_penalty': actual_penalty,
                                's_final': s_final
                            })
                        except:
                            pass
                    i += 1
        
        i += 1
    
    if current_query:
        queries.append(current_query)
    
    return queries

def extract_type1_bad_cases(queries):
    """
    第一类坏例：MAP 血崩的元凶（被大锤砸死的好文）
    条件: relevance > 0, Actual_Penalty > 0.02, Rank(Changed) > Rank(OG) + 20
    """
    bad_cases = []
    
    for query in queries:
        # 创建 OG 排名映射
        og_rank_map = {doc['doc_id']: doc['rank'] for doc in query['og_docs']}
        
        # 检查被惩罚的文档
        for doc in query['penalized_docs']:
            if doc['relevance'] > 0 and doc['actual_penalty'] > 0.02:
                og_rank = og_rank_map.get(doc['doc_id'], doc['rank'])
                rank_drop = doc['rank'] - og_rank
                
                if rank_drop > 20:
                    bad_cases.append({
                        'query_id': query['query_id'],
                        'neg_words': query['neg_words'],
                        'doc_id': doc['doc_id'],
                        'relevance': doc['relevance'],
                        'og_rank': og_rank,
                        'changed_rank': doc['rank'],
                        'rank_drop': rank_drop,
                        's_pos': doc['s_pos'],
                        's_neg_proj': doc['s_neg_proj'],
                        'dynamic_tau': doc['dynamic_tau'],
                        'actual_penalty': doc['actual_penalty'],
                        's_final': doc['s_final']
                    })
    
    # 按排名下降幅度排序
    bad_cases.sort(key=lambda x: x['rank_drop'], reverse=True)
    return bad_cases

def extract_type2_bad_cases(queries):
    """
    第二类坏例：pMRR 停滞的元凶（免疫惩罚的铁头烂文）
    条件: relevance == 0, Contains_Negative_Word == True, Actual_Penalty == 0, Rank(Changed) <= 20
    """
    bad_cases = []
    
    for query in queries:
        # 漏网烂文已经在白盒报告中识别
        for doc in query['escaped_docs']:
            if doc['relevance'] == 0 and doc['actual_penalty'] < 0.005 and doc['rank'] <= 20:
                bad_cases.append({
                    'query_id': query['query_id'],
                    'neg_words': query['neg_words'],
                    'doc_id': doc['doc_id'],
                    'relevance': doc['relevance'],
                    'changed_rank': doc['rank'],
                    's_pos': doc['s_pos'],
                    's_neg_proj': doc['s_neg_proj'],
                    'dynamic_tau': doc['dynamic_tau'],
                    'over_threshold': doc['over_threshold'],
                    'actual_penalty': doc['actual_penalty'],
                    's_final': doc['s_final']
                })
    
    # 按排名排序
    bad_cases.sort(key=lambda x: x['changed_rank'])
    return bad_cases

def main():
    print("=" * 80)
    print("🔍 深度 Error Analysis")
    print("=" * 80)
    
    # 解析白盒报告
    report_path = Path('evaluation/dsclr/lap_eval/4.4-delta-grid-search/whitebox_report.md')
    queries = parse_whitebox_report(report_path)
    
    print(f"\n✅ 解析了 {len(queries)} 个查询的白盒报告")
    
    # 提取第一类坏例
    print("\n" + "=" * 80)
    print("🚨 第一类坏例：MAP 血崩的元凶（被大锤砸死的好文）")
    print("=" * 80)
    print("筛选条件: relevance > 0, Actual_Penalty > 0.02, Rank(Changed) > Rank(OG) + 20")
    
    type1_cases = extract_type1_bad_cases(queries)
    print(f"\n找到 {len(type1_cases)} 个第一类坏例")
    
    if type1_cases:
        print("\n前 3 个最典型的例子:")
        for i, case in enumerate(type1_cases[:3], 1):
            print(f"\n{'='*80}")
            print(f"案例 {i}:")
            print(f"  Query ID: {case['query_id']}")
            print(f"  负向词: {case['neg_words']}")
            print(f"  Doc ID: {case['doc_id']}")
            print(f"  Relevance: {case['relevance']}")
            print(f"  OG 排名: {case['og_rank']}")
            print(f"  Changed 排名: {case['changed_rank']}")
            print(f"  排名下降: {case['rank_drop']}")
            print(f"  S_pos: {case['s_pos']:.4f}")
            print(f"  S_neg_proj: {case['s_neg_proj']:.4f}")
            print(f"  Dynamic_Tau: {case['dynamic_tau']:.4f}")
            print(f"  Actual_Penalty: {case['actual_penalty']:.4f}")
            print(f"  S_final: {case['s_final']:.4f}")
    
    # 提取第二类坏例
    print("\n" + "=" * 80)
    print("🚨 第二类坏例：pMRR 停滞的元凶（免疫惩罚的铁头烂文）")
    print("=" * 80)
    print("筛选条件: relevance == 0, Actual_Penalty < 0.005, Rank(Changed) <= 20")
    
    type2_cases = extract_type2_bad_cases(queries)
    print(f"\n找到 {len(type2_cases)} 个第二类坏例")
    
    if type2_cases:
        print("\n前 3 个最典型的例子:")
        for i, case in enumerate(type2_cases[:3], 1):
            print(f"\n{'='*80}")
            print(f"案例 {i}:")
            print(f"  Query ID: {case['query_id']}")
            print(f"  负向词: {case['neg_words']}")
            print(f"  Doc ID: {case['doc_id']}")
            print(f"  Relevance: {case['relevance']}")
            print(f"  Changed 排名: {case['changed_rank']}")
            print(f"  S_pos: {case['s_pos']:.4f}")
            print(f"  S_neg_proj: {case['s_neg_proj']:.4f}")
            print(f"  Dynamic_Tau: {case['dynamic_tau']:.4f}")
            print(f"  Over_Threshold: {case['over_threshold']:.4f}")
            print(f"  Actual_Penalty: {case['actual_penalty']:.4f}")
            print(f"  S_final: {case['s_final']:.4f}")
    
    # 保存结果
    output = {
        'type1_cases': type1_cases[:3],
        'type2_cases': type2_cases[:3]
    }
    
    output_path = Path('evaluation/dsclr/lap_eval/4.4-delta-grid-search/deep_error_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
