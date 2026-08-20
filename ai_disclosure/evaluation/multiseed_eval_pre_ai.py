# -*- coding: utf-8 -*-
"""
ARCHIVED PRE-AI-ASSISTED VERSION.

This is the original notebook fragment (Google Colab) that computed the
multi-seed, size-stratified recall statistics underlying Tables 3 and 4 and
the operational assessment rule, exactly as authored by the researchers
before any AI-assisted repackaging. It is extracted verbatim (Chinese
print-string comments and hardcoded Colab paths included) from the authors'
working notebook script, where it relied on notebook-level context defined
in earlier cells:

    OUT    = '/content/ldpolyp'                      # dataset root
    P33, P67 = 0.0187, 0.0631                         # frozen thresholds
    BACKUP = '/content/drive/MyDrive/small/LDPolyp_backup'
    seeds  = {'42': (baseline_ckpt, p2_ckpt), ..., '49': (...)}
    from scipy import stats
    import os, glob, numpy as np, json
    from ultralytics import YOLO

The post-AI-assisted version (`evaluation/multiseed_eval.py`) repackages this
logic into a standalone CLI script (argparse, generalised dataset/weights
paths, English print output) and fixes one bug that was present in a later,
separately maintained copy of this code that had been pushed to the GitHub
repository before this notebook version was rediscovered: that copy computed
the seed-to-seed standard deviation by pooling all 16 per-seed values
together (giving SD_seed = 3.24), instead of the pooled standard deviation
of the two per-configuration SDs defined in Section 3.1 of the paper,
sqrt((SD_B^2 + SD_C^2) / 2) (giving SD_seed = 3.32-3.33, matching Table 3).
The statistical logic below -- box_iou, evalsize, and stat_block, including
the pooled-SD formula -- already uses the correct definition and matches the
numbers reported in the paper.
"""

# --- notebook-level context this fragment depended on (see docstring above) ---
# OUT, P33, P67, BACKUP, seeds, and the imports below were defined in earlier
# cells of the same notebook session and are not redefined here, to keep this
# archived fragment faithful to what was actually run.

def box_iou(b,bs):
    if len(bs)==0: return np.array([])
    x1=np.maximum(b[0],bs[:,0]);y1=np.maximum(b[1],bs[:,1]);x2=np.minimum(b[2],bs[:,2]);y2=np.minimum(b[3],bs[:,3])
    iw=np.clip(x2-x1,0,None);ih=np.clip(y2-y1,0,None);inter=iw*ih
    a1=(b[2]-b[0])*(b[3]-b[1]);a2=(bs[:,2]-bs[:,0])*(bs[:,3]-bs[:,1])
    return inter/(a1+a2-inter+1e-9)

def evalsize(w,imgsz=1280,conf=0.25,iou_m=0.5):
    m=YOLO(w);hit={'small':0,'medium':0,'large':0};tot={'small':0,'medium':0,'large':0}
    for ip in sorted(glob.glob(f'{OUT}/images/test/*.jpg')):
        lp=f"{OUT}/labels/test/{os.path.basename(ip)[:-4]}.txt";gts=[]
        if os.path.exists(lp):
            for ln in open(lp):
                p=ln.split()
                if len(p)==5:_,cx,cy,w2,h=map(float,p);gts.append([cx,cy,w2,h])
        if not gts: continue
        r=m.predict(ip,conf=conf,imgsz=imgsz,verbose=False)[0]
        W,H=r.orig_shape[1],r.orig_shape[0]
        pr=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
        for cx,cy,w2,h in gts:
            ra=w2*h;bk='small' if ra<P33 else('medium' if ra<P67 else 'large');tot[bk]+=1
            g=[(cx-w2/2)*W,(cy-h/2)*H,(cx+w2/2)*W,(cy+h/2)*H];io=box_iou(g,pr)
            if len(io)>0 and io.max()>=iou_m:hit[bk]+=1
    return {k:hit[k]/tot[k] if tot[k]>0 else 0 for k in tot}

data={'baseline':{'small':[],'medium':[],'large':[]},'p2':{'small':[],'medium':[],'large':[]}}
print("开始评估8个种子...\n")
for s,(bw,pw) in seeds.items():
    rb=evalsize(bw); rp=evalsize(pw)
    for k in ['small','medium','large']:
        data['baseline'][k].append(rb[k]); data['p2'][k].append(rp[k])
    print(f"seed{s}: base小={rb['small']*100:.1f}% p2小={rp['small']*100:.1f}% (差{(rp['small']-rb['small'])*100:+.1f})")

def stat_block(k):
    b=np.array(data['baseline'][k])*100; p=np.array(data['p2'][k])*100
    t,pval=stats.ttest_rel(p,b)
    diff=p.mean()-b.mean()
    sd_pool=np.sqrt((b.std(ddof=1)**2+p.std(ddof=1)**2)/2)
    d=diff/sd_pool if sd_pool>0 else 0
    return b,p,diff,pval,sd_pool,d

print("\n"+"="*60)
print(f"{'档':<8}{'baseline':<16}{'p2':<16}{'diff':<8}{'p':<8}")
print("="*60)
for k in ['small','medium','large']:
    b,p,diff,pval,sd,d=stat_block(k)
    print(f"{k:<8}{b.mean():.1f}±{b.std(ddof=1):.1f}      {p.mean():.1f}±{p.std(ddof=1):.1f}      {diff:+.2f}   {pval:.3f}")

b,p,diff,pval,sd,d=stat_block('small')
print("\n"+"="*60)
print("小息肉 Operational Assessment Rule（8种子）")
print("="*60)
print(f"  Delta_small = {diff:+.2f}pp")
print(f"  SD_seed(pooled) = {sd:.2f}")
print(f"  p值 = {pval:.4f}")
print(f"  Cohen's d = {d:.3f} (descriptive)")
print(f"\n  条件1 Delta>0:        {'满足' if diff>0 else '不满足'}")
print(f"  条件2 p<0.05:         {'满足' if pval<0.05 else '不满足'}")
print(f"  条件3 Delta>SD_seed:  {'满足' if diff>sd else '不满足'}")
verdict = "可靠改进" if (diff>0 and pval<0.05 and diff>sd) else "不能判定为可靠改进"
print(f"\n  → 判定: {verdict}")

out = {
  'n_seeds': 8, 'seeds': list(seeds.keys()),
  'data': {kk:{k:[float(x) for x in data[kk][k]] for k in data[kk]} for kk in data},
  'small': {'baseline_mean':float(b.mean()),'baseline_std':float(b.std(ddof=1)),
            'p2_mean':float(p.mean()),'p2_std':float(p.std(ddof=1)),
            'mean_diff':float(diff),'pooled_sd':float(sd),'pval':float(pval),'cohens_d':float(d)}
}
os.makedirs(f'{BACKUP}/results', exist_ok=True)
with open(f'{BACKUP}/results/multiseed_8seeds.json','w') as f:
    json.dump(out,f,indent=2,ensure_ascii=False)
print(f"\n结果已存: {BACKUP}/results/multiseed_8seeds.json")
