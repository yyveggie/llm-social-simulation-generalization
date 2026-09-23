#!/usr/bin/env python
import csv, json, os, sys, glob
csv.field_size_limit(10**7)
if len(sys.argv) < 2:
    sys.exit("usage: python apply_opus_judge_overrides.py <overrides.json>")
OV = sys.argv[1]
ovs = json.load(open(OV))

byrun = {}
for o in ovs:
    byrun.setdefault(o['run'], {})[str(o['indexx'])] = o
patched_total = 0
for run, idxmap in byrun.items():
    p = os.path.join(run, "parsed.csv")
    if not os.path.exists(p): 
        print(f"[skip] 无 {p}"); continue
    rows = list(csv.DictReader(open(p)))
    fields = rows[0].keys()
    n = 0
    for r in rows:
        o = idxmap.get(str(r.get('indexx')))
        if not o: continue
        if r.get('parse_path') != 'fallback_first':
            continue
        exp = o['exp']; upper = 100.0 if exp=="battery_life" else float(o['stakes'])
        if o['kind'] == 'NUM':
            v = float(o['value'])
            r['decision'] = repr(v) if False else (str(int(v)) if v==int(v) else str(v))
            r['propshare'] = str(v/upper) if upper else ''
            r['unusable_reason'] = ''
            r['parse_path'] = 'judge_reparse'
        elif o['kind'] == 'REFUSE':
            r['decision'] = ''; r['propshare'] = ''
            r['unusable_reason'] = 'refusal'; r['parse_path'] = ''
        else:
            r['decision'] = ''; r['propshare'] = ''
            r['unusable_reason'] = 'ambiguous'; r['parse_path'] = ''
        n += 1
    if n:
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(rows)
        os.utime(p, None)
        patched_total += n
        print(f"[ok] {run.split('claude-opus-4-8/')[-1].rstrip('/'):40s} 覆盖 {n} 行")
print(f"\n合计覆盖 {patched_total} 行 opus fallback。")
