"""E-V0 GATE: reproduce Sikka et al. 2019 (J Neurosci 39:4775) on REM_Turku.

Published claim: FAA = ln[F4]-ln[F3] on alpha CSD power predicts dream anger.
Sikka's unit of analysis is the PARTICIPANT (N=17). Anger = mDES NA1
(Angry/Irritated/Annoyed); interest = PA6 (Interested/Alert/Curious).

Result 2026-08-25: PARTIAL PASS. Descriptives match (anger in 40% of dreams vs
41% published; interest 88% vs 88%; 119 clean epochs vs 114.45). Effect has the
right sign and frontal topography (F4-F3 rho=+0.335) but p=0.19 at N=17.
At AWAKENING level the effect vanishes (rho=+0.040), i.e. Sikka's finding is a
between-subject TRAIT effect, not a within-subject STATE effect.

Usage: python faa_gate.py --prepared remturku_prepared.json --zip REM_Turku.zip
"""
import argparse, csv, io, json, zipfile, statistics as st, math, random
from collections import defaultdict

def pear(x,y):
    mx,my=st.mean(x),st.mean(y)
    d=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/d if d else float('nan')

def rank(v):
    s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and v[s[j+1]]==v[s[i]]: j+=1
        for k in range(i,j+1): r[s[k]]=(i+j)/2+1
        i=j+1
    return r

def spearman(x,y): return pear(rank(x),rank(y))

def perm_p(x,y,fn,n=20000,seed=0):
    rnd=random.Random(seed); obs=fn(x,y); yy=list(y); c=0
    for _ in range(n):
        rnd.shuffle(yy)
        if abs(fn(x,yy))>=abs(obs): c+=1
    return (c+1)/(n+1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',required=True); ap.add_argument('--zip',required=True)
    ap.add_argument('--out',default='faa_gate_rows.json'); a=ap.parse_args()
    prep=json.load(open(a.prepared))['records']
    z=zipfile.ZipFile(a.zip)
    rat={r['Filename']:r for r in csv.DictReader(io.StringIO(z.read('REM_Turku/Data/Ratings.csv').decode('utf-8-sig')))}
    rec={r['Filename']:r for r in csv.DictReader(io.StringIO(z.read('REM_Turku/Records.csv').decode('utf-8-sig')))}
    def f(x):
        try: return float(x)
        except: return None
    rows=[]
    for p in prep:
        fn=p['filename']
        if fn not in rat or fn not in rec: continue
        ang=f(rat[fn].get('SR_NA1'))
        if ang is None: continue
        rows.append({'file':fn,'subj':rec[fn]['Subject ID'],'anger':ang,
                     'interest':f(rat[fn].get('SR_PA6')),
                     **{k:v for k,v in p.items() if k.startswith('faa_')}})
    print(f"awakenings with FAA + anger: {len(rows)}  subjects: {len(set(r['subj'] for r in rows))}")
    bys=defaultdict(list)
    for r in rows: bys[r['subj']].append(r)
    keys=sorted(bys)
    sub_faa=[st.mean(x['faa_F4-F3'] for x in bys[s]) for s in keys]
    sub_ang=[st.mean(x['anger'] for x in bys[s]) for s in keys]
    sub_int=[st.mean(x['interest'] for x in bys[s] if x['interest'] is not None) for s in keys]
    print(f"\n=== PARTICIPANT LEVEL (Sikka design, N={len(keys)}) ===")
    for nm,y in (('anger',sub_ang),('interest',sub_int)):
        print(f"  FAA(F4-F3) vs {nm:<9} r={pear(sub_faa,y):+.3f} rho={spearman(sub_faa,y):+.3f} "
              f"perm p={perm_p(sub_faa,y,spearman):.4f}")
    print(f"  anger>0 in {sum(1 for r in rows if r['anger']>0)/len(rows):.0%} of dreams (Sikka: 41%)")
    print(f"  interest>0 in {sum(1 for r in rows if r['interest'] and r['interest']>0)/len(rows):.0%} (Sikka: 88%)")
    print(f"\n=== SPATIAL SPECIFICITY vs anger ===")
    for k in [k for k in rows[0] if k.startswith('faa_')]:
        v=[st.mean(x[k] for x in bys[s]) for s in keys]
        print(f"  {k.replace('faa_',''):<9} rho={spearman(v,sub_ang):+.3f} p={perm_p(v,sub_ang,spearman,5000):.4f}")
    x=[r['faa_F4-F3'] for r in rows]; y=[r['anger'] for r in rows]
    print(f"\n=== AWAKENING LEVEL (n={len(rows)}) ===")
    print(f"  FAA vs anger r={pear(x,y):+.3f} rho={spearman(x,y):+.3f} perm p={perm_p(x,y,spearman):.4f}")
    json.dump(rows,open(a.out,'w'),indent=1)

if __name__=='__main__': main()
