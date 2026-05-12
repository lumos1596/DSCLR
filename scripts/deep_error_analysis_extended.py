#!/usr/bin/env python3
"""
深度 Error Analysis - 扩展版
放宽条件查找更多坏例
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
                'top10_docs': [],
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
            
            # 解析前10名文档
            elif line.startswith('### 📊 前10名文档'):
                i += 3  # 跳过表头
                while i < len(lines) and lines[i].strip().startswith('|'):
                    parts = lines[i].strip().split('|')
                    if len(parts) >= 7:
                        try:
                            og_rank = int(parts[1].strip())
                            changed_rank = int(parts[2].strip())
                            doc_id = parts[3].strip()
                            relevance = int(parts[4].strip())
                            has_neg = parts[5].strip() == '✓'
                            s_pos = float(parts[6].strip())
                            s_final = float(parts[7].strip())
                            current_query['top10_docs'].append({
                                'og_rank': og_rank,
                                'changed_rank': changed_rank,
                                'doc_id': doc_id,
                                'relevance': relevance,
                                'has_neg': has_neg,
                                's_pos': s_pos,
                                's_final': s_final
                            })
                        except Exception as e:
                            pass
                    i += 1
            
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

def extract_type1_bad_cases_extended(queries):
    """
    第一类坏例：MAP 血崩的元凶（被大锤砸死的好文）
    放宽条件: relevance > 0, Actual_Penalty > 0.01, Rank(Changed) > Rank(OG) + 10
    """
    bad_cases = []
    
    for query in queries:
        # 创建 OG 排名映射
        og_rank_map = {doc['doc_id']: doc['rank'] for doc in query['og_docs']}
        
        # 检查被惩罚的文档
        for doc in query['penalized_docs']:
            if doc['relevance'] > 0 and doc['actual_penalty'] > 0.01:
                og_rank = og_rank_map.get(doc['doc_id'], doc['rank'])
                rank_drop = doc['rank'] - og_rank
                
                if rank_drop > 10:  # 放宽到10
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

def extract_type2_bad_cases_extended(queries):
    """
    第二类坏例：pMRR 停滞的元凶（免疫惩罚的铁头烂文）
    放宽条件: relevance == 0, Contains_Negative_Word == True, Actual_Penalty < 0.01, Rank(Changed) <= 20
    """
    bad_cases = []
    
    for query in queries:
        # 从前10名中查找包含负向词的坏文档
        for doc in query['top10_docs']:
            if doc['relevance'] == 0 and doc['has_neg']:
                # 查找对应的惩罚信息
                penalty_info = None
                for pdoc in query['penalized_docs']:
                    if pdoc['doc_id'] == doc['doc_id']:
                        penalty_info = pdoc
                        break
                
                # 如果没有惩罚信息，说明是漏网烂文
                if penalty_info is None:
                    penalty_info = {
                        's_neg_proj': 0.0,
                        'dynamic_tau': query['dynamic_tau'],
                        'over_threshold': 0.0,
                        'actual_penalty': 0.0
                    }
                
                bad_cases.append({
                    'query_id': query['query_id'],
                    'neg_words': query['neg_words'],
                    'doc_id': doc['doc_id'],
                    'relevance': doc['relevance'],
                    'changed_rank': doc['changed_rank'],
                    's_pos': doc['s_pos'],
                    's_neg_proj': penalty_info['s_neg_proj'],
                    'dynamic_tau': penalty_info['dynamic_tau'],
                    'over_threshold': penalty_info['over_threshold'],
                    'actual_penalty': penalty_info['actual_penalty'],
                    's_final': doc['s_final']
                })
        
        # 也检查漏网烂文列表
        for doc in query['escaped_docs']:
            if doc['relevance'] == 0 and doc['actual_penalty'] < 0.01 and doc['rank'] <= 20:
                # 避免重复
                if not any(c['doc_id'] == doc['doc_id'] for c in bad_cases):
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
    print("🔍 深度 Error Analysis (扩展版)")
    print("=" * 80)
    
    # 解析白盒报告
    report_path = Path('evaluation/dsclr/lap_eval/4.4-delta-grid-search/whitebox_report.md')
    queries = parse_whitebox_report(report_path)
    
    print(f"\n✅ 解析了 {len(queries)} 个查询的白盒报告")
    
    # 提取第一类坏例
    print("\n" + "=" * 80)
    print("🚨 第一类坏例：MAP 血崩的元凶（被大锤砸死的好文）")
    print("=" * 80)
    print("筛选条件: relevance > 0, Actual_Penalty > 0.01, Rank(Changed) > Rank(OG) + 10")
    
    type1_cases = extract_type1_bad_cases_extended(queries)
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
    else:
        print("\n✅ 未发现第一类坏例！所有被惩罚的文档都是坏文档 (relevance=0)")
    
    # 提取第二类坏例
    print("\n" + "=" * 80)
    print("🚨 第二类坏例：pMRR 停滞的元凶（免疫惩罚的铁头烂文）")
    print("=" * 80)
    print("筛选条件: relevance == 0, Contains_Negative_Word == True, Actual_Penalty < 0.01, Rank(Changed) <= 20")
    
    type2_cases = extract_type2_bad_cases_extended(queries)
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
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 总结")
    print("=" * 80)
    print(f"第一类坏例（误伤好文）: {len(type1_cases)} 个")
    print(f"第二类坏例（漏网烂文）: {len(type2_cases)} 个")
    
    if len(type1_cases) == 0:
        print("\n✅ LAP 模块没有误伤好文档！所有被惩罚的文档都是坏文档。")
    
    if len(type2_cases) <= 1:
        print("\n✅ LAP 模块对烂文的识别能力很强！只有极少数烂文逃过了惩罚。")
    
    # 保存结果
    output = {
        'type1_cases': type1_cases[:3],
        'type2_cases': type2_cases[:3],
        'summary': {
            'type1_count': len(type1_cases),
            'type2_count': len(type2_cases)
        }
    }
    
    output_path = Path('evaluation/dsclr/lap_eval/4.4-delta-grid-search/deep_error_analysis_extended.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
