"""Analyze V5 patterns for prompt engineering design"""
import json, datasets, re
from collections import Counter

def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r['qid']] = r
    return records

v5_core17 = load_jsonl('dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_Core17InstructionRetrieval.jsonl')
v5_robust04 = load_jsonl('dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_Robust04InstructionRetrieval.jsonl')
v5_news21 = load_jsonl('dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_News21InstructionRetrieval.jsonl')

inst_data = {}
for ds_name, hf_path in [('Core17', 'jhu-clsp/core17-instructions-mteb'),
                           ('Robust04', 'jhu-clsp/robust04-instructions-mteb'),
                           ('News21', 'jhu-clsp/news21-instructions-mteb')]:
    ds_inst = datasets.load_dataset(hf_path, 'instruction')
    for item in list(ds_inst.values())[0]:
        qid = str(item.get('query-id', ''))
        inst_data[qid] = str(item.get('instruction', ''))

all_v5 = {**v5_core17, **v5_robust04, **v5_news21}

exclusion_types = Counter()
for qid, r in all_v5.items():
    inst = inst_data.get(qid, '')
    qm = r['q_minus']
    if qm == '[NONE]':
        exclusion_types['no_exclusion'] += 1
    else:
        has_while = bool(re.search(r'while\s+.+?(?:irrelevant|not\s+relevant)', inst, re.IGNORECASE))
        has_irrelevant = bool(re.search(r'(?:is|are)\s+irrelevant', inst, re.IGNORECASE))
        has_not_relevant = bool(re.search(r'(?:is|are)\s+not\s+relevant', inst, re.IGNORECASE))
        has_nor = bool(re.search(r'\bnor\b', inst, re.IGNORECASE))
        has_do_not = bool(re.search(r'do\s+not\s+(?:include|consider|count)', inst, re.IGNORECASE))
        if has_while:
            exclusion_types['contrast_exclusion'] += 1
        elif has_irrelevant:
            exclusion_types['explicit_irrelevant'] += 1
        elif has_not_relevant:
            exclusion_types['explicit_not_relevant'] += 1
        elif has_nor:
            exclusion_types['nor_pattern'] += 1
        elif has_do_not:
            exclusion_types['do_not'] += 1
        else:
            exclusion_types['scope_limitation'] += 1

print("1. Exclusion pattern distribution:")
for k, v in sorted(exclusion_types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/len(all_v5)*100:.1f}%)")

qm_lens = [len(r['q_minus']) for r in all_v5.values() if r['q_minus'] != '[NONE]']
print(f"\n2. Q_minus length (non-NONE): min={min(qm_lens)}, max={max(qm_lens)}, mean={sum(qm_lens)/len(qm_lens):.1f}")

qp_lens = [len(r['q_plus']) for r in all_v5.values()]
print(f"3. Q_plus length: min={min(qp_lens)}, max={max(qp_lens)}, mean={sum(qp_lens)/len(qp_lens):.1f}")

print("\n4. [NONE] rate by dataset:")
for name, data in [('Core17', v5_core17), ('Robust04', v5_robust04), ('News21', v5_news21)]:
    none_count = sum(1 for r in data.values() if r['q_minus'] == '[NONE]')
    print(f"  {name}: {none_count}/{len(data)} ({none_count/len(data)*100:.1f}%)")

item_counts = []
for r in all_v5.values():
    if r['q_minus'] != '[NONE]':
        items = [x.strip() for x in r['q_minus'].split(',') if x.strip()]
        item_counts.append(len(items))
counter = Counter(item_counts)
print("\n5. Q_minus item count distribution:")
for k in sorted(counter.keys()):
    print(f"  {k} items: {counter[k]} ({counter[k]/len(item_counts)*100:.1f}%)")

print("\n6. Scope limitation cases (V5 has exclusion but no explicit 'not relevant'):")
scope_cases = 0
for qid, r in all_v5.items():
    if r['q_minus'] == '[NONE]':
        continue
    inst = inst_data.get(qid, '')
    has_explicit = bool(re.search(r'(?:is|are)\s+(?:not\s+)?relevant|irrelevant|do\s+not|nor\s+are', inst, re.IGNORECASE))
    if not has_explicit:
        scope_cases += 1
        if scope_cases <= 8:
            print(f"\n  QID: {qid}")
            print(f"  Inst: {inst[:200]}")
            print(f"  V5 Q_minus: {r['q_minus']}")
print(f"\n  Total scope limitation cases: {scope_cases}")
