"""Project REM_Turku mDES self-ratings into continuous VAD via NRC-VAD (Mohammad, ACL 2018).

VAD(awakening) = sum_i SR_i * VAD(item_i) / sum_i SR_i
Each mDES item is a synonym triplet; item VAD = mean of its terms' NRC-VAD vectors.

Result on the 2026-08-25 run: 115 of 122 rated awakenings yield a VAD vector,
114 join to Records.csv, 17 subjects. Valence and Dominance correlate at
r = 0.933 at awakening level and r = 0.923 at ITEM level, so the collinearity is
a property of the mDES item set, not an averaging artifact: the 10 positive items
span dominance 0.499-0.783 and the 10 negative items 0.186-0.413, with zero
overlap. Hence the usable target space is 2D (valence, arousal), not 3D.

Usage: python mdes_to_vad.py --zip REM_Turku.zip --nrc NRC-VAD-Lexicon.txt --out vad.json
"""
import argparse, csv, io, json, zipfile, statistics as st, math

MDES = {
 'PA1':['amused','funloving','giggly'],       'PA2':['awe','wonder','amazement'],
 'PA3':['grateful','appreciative','thankful'],'PA4':['hopeful','optimistic','encouraged'],
 'PA5':['inspired','uplifted','elevated'],    'PA6':['interested','alert','curious'],
 'PA7':['joyful','glad','happy'],             'PA8':['love','closeness','trust'],
 'PA9':['proud','confident','selfassured'],   'PA10':['serene','content','peaceful'],
 'NA1':['angry','irritated','annoyed'],       'NA2':['ashamed','humiliated','disgraced'],
 'NA3':['contemptuous','scornful','disdainful'],'NA4':['disgust','distaste','revulsion'],
 'NA5':['embarrassed','selfconscious','blushing'],'NA6':['guilty','repentant','blameworthy'],
 'NA7':['hate','distrust','suspicion'],       'NA8':['sad','downhearted','unhappy'],
 'NA9':['scared','fearful','afraid'],         'NA10':['stressed','nervous','overwhelmed'],
}
ALIAS = {'funloving':'fun','selfassured':'assured','selfconscious':'conscious'}
ORDER = [f'PA{i}' for i in range(1,11)] + [f'NA{i}' for i in range(1,11)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',required=True); ap.add_argument('--nrc',required=True)
    ap.add_argument('--out',required=True); a=ap.parse_args()
    lex={}
    for line in open(a.nrc,encoding='utf-8'):
        p=line.rstrip('\n').split('\t')
        if len(p)==4: lex[p[0]]=(float(p[1]),float(p[2]),float(p[3]))
    item_vad={}; misses=[]
    for item,terms in MDES.items():
        vecs=[]
        for t in terms:
            v=lex.get(t) or lex.get(ALIAS.get(t,''))
            (vecs.append(v) if v else misses.append((item,t)))
        if vecs: item_vad[item]=tuple(sum(v[k] for v in vecs)/len(vecs) for k in range(3))
    print("mDES item -> NRC-VAD:")
    for k in ORDER:
        v=item_vad[k]; print(f"  {k:<5} {'/'.join(MDES[k]):<38} V={v[0]:.3f} A={v[1]:.3f} D={v[2]:.3f}")
    print(f"term misses ({len(misses)}): {misses}")

    z=zipfile.ZipFile(a.zip)
    rows=list(csv.DictReader(io.StringIO(z.read('REM_Turku/Data/Ratings.csv').decode('utf-8-sig'))))
    rec ={r['Filename']:r for r in csv.DictReader(io.StringIO(z.read('REM_Turku/Records.csv').decode('utf-8-sig')))}
    out=[]
    for r in rows:
        w=[]
        for item in ORDER:
            try: s=float(r[f'SR_{item}'])
            except: s=0.0
            if s>0 and item in item_vad: w.append((s,item_vad[item]))
        if not w: continue
        tot=sum(s for s,_ in w)
        vad=tuple(sum(s*v[k] for s,v in w)/tot for k in range(3))
        f=r['Filename']
        out.append({'filename':f,'subject':rec.get(f,{}).get('Subject ID'),
                    'intensity':tot,'valence':vad[0],'arousal':vad[1],'dominance':vad[2],
                    'in_records':f in rec})
    def pear(x,y):
        mx,my=st.mean(x),st.mean(y)
        return sum((p-mx)*(q-my) for p,q in zip(x,y))/math.sqrt(
            sum((p-mx)**2 for p in x)*sum((q-my)**2 for q in y))
    V=[o['valence'] for o in out]; A=[o['arousal'] for o in out]; D=[o['dominance'] for o in out]
    print(f"\nVAD computed: {len(out)}  joined: {sum(1 for o in out if o['in_records'])}  "
          f"subjects: {len(set(o['subject'] for o in out if o['subject']))}")
    for nm,v in (('valence',V),('arousal',A),('dominance',D)):
        print(f"  {nm:<10} mean={st.mean(v):.3f} sd={st.pstdev(v):.3f} range={max(v)-min(v):.3f}")
    print(f"  corr(V,D)={pear(V,D):+.3f}  corr(V,A)={pear(V,A):+.3f}  corr(A,D)={pear(A,D):+.3f}")
    iv=[item_vad[k] for k in ORDER]
    print(f"  ITEM-level corr(V,D)={pear([x[0] for x in iv],[x[2] for x in iv]):+.3f}  "
          f"(collinearity is in the instrument, not the averaging)")
    json.dump(out,open(a.out,'w'),indent=1); print(f"-> {a.out}")

if __name__=='__main__': main()
