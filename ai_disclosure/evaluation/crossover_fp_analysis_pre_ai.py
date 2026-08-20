# -*- coding: utf-8 -*-
"""
ARCHIVED PRE-AI-ASSISTED VERSION.

Original notebook fragment (Google Colab) that computed the per-seed
crossover locations and small-scale false-positive changes underlying
Section 5.2 and Table 5, extracted verbatim from the authors' working
notebook script. It relied on notebook-level context defined in earlier
cells:

    OUT    = '/content/ldpolyp'
    BACKUP = '/content/drive/MyDrive/small/LDPolyp_backup'
    seeds  = {'42': (baseline_ckpt, p2_ckpt), ..., '49': (...)}
    import os, glob, json, numpy as np
    from ultralytics import YOLO

The post-AI-assisted version (`evaluation/crossover_fp_analysis.py`)
repackages this logic into a standalone CLI script (argparse, generalised
paths, English print output, `--out` option). The algorithm itself --
box_iou, eval_full, sliding_recall, and find_crossover, including the fixed
0.3-30% / 40-point logarithmic grid, the descending-crossing-only crossover
rule, and the frozen p33/p67 false-positive binning thresholds -- is
unchanged from this original version; the AI-assisted repackaging was
verified against it with 200 randomised equivalence trials before being
adopted into the repository.
"""

# --- notebook-level context this fragment depended on (see docstring above) ---

def box_iou(b,bs):
    if len(bs)==0: return np.array([])
    x1=np.maximum(b[0],bs[:,0]);y1=np.maximum(b[1],bs[:,1]);x2=np.minimum(b[2],bs[:,2]);y2=np.minimum(b[3],bs[:,3])
    iw=np.clip(x2-x1,0,None);ih=np.clip(y2-y1,0,None);inter=iw*ih
    a1=(b[2]-b[0])*(b[3]-b[1]);a2=(bs[:,2]-bs[:,0])*(bs[:,3]-bs[:,1])
    return inter/(a1+a2-inter+1e-9)

# 返回每个GT的(相对面积, 是否命中) + 总FP数（按尺度分）
def eval_full(w, imgsz=1280, conf=0.25, iou_m=0.5):
    m=YOLO(w); recs=[]
    fp_small=fp_med=fp_large=0
    for ip in sorted(glob.glob(f'{OUT}/images/test/*.jpg')):
        lp=f"{OUT}/labels/test/{os.path.basename(ip)[:-4]}.txt"; gts=[]
        if os.path.exists(lp):
            for ln in open(lp):
                p=ln.split()
                if len(p)==5:_,cx,cy,w2,h=map(float,p);gts.append([cx,cy,w2,h])
        r=m.predict(ip,conf=conf,imgsz=imgsz,verbose=False)[0]
        W,H=r.orig_shape[1],r.orig_shape[0]
        preds=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
        gt_xyxy=[]
        for cx,cy,w2,h in gts:
            rel=w2*h
            g=[(cx-w2/2)*W,(cy-h/2)*H,(cx+w2/2)*W,(cy+h/2)*H]
            gt_xyxy.append(g)
            io=box_iou(g,preds)
            hit=1 if (len(io)>0 and io.max()>=iou_m) else 0
            recs.append((rel,hit))
        # FP统计：每个预测框，若与所有GT的IoU都<阈值，算FP，按其面积分档
        for pb in preds:
            pa=(pb[2]-pb[0])*(pb[3]-pb[1])/(W*H)
            matched=False
            for g in gt_xyxy:
                if box_iou(g,np.array([pb]))[0]>=iou_m: matched=True; break
            if not matched:
                if pa<0.0187: fp_small+=1
                elif pa<0.0631: fp_med+=1
                else: fp_large+=1
    return np.array(recs), (fp_small,fp_med,fp_large)

def sliding_recall(recs, xs, win=0.5):
    rel=recs[:,0]*100; hit=recs[:,1]; ys=[]
    for x in xs:
        lo,hi=x/(1+win),x*(1+win)
        mask=(rel>=lo)&(rel<hi)
        ys.append(hit[mask].mean()*100 if mask.sum()>=10 else np.nan)
    return np.array(ys)

def find_crossover(yb, yp, xs):
    diff=yp-yb; valid=~np.isnan(diff)
    xv=xs[valid]; dv=diff[valid]
    for i in range(len(dv)-1):
        if dv[i]>0 and dv[i+1]<=0:
            x0,x1=xv[i],xv[i+1]; d0,d1=dv[i],dv[i+1]
            return x0+(x1-x0)*d0/(d0-d1)
    return None

xs=np.logspace(np.log10(0.3),np.log10(30),40)
crossovers=[]; fp_changes=[]
print("评估8种子的crossover和FP...\n")
for s,(bw,pw) in seeds.items():
    rb, fpb = eval_full(bw)
    rp, fpp = eval_full(pw)
    yb=sliding_recall(rb,xs); yp=sliding_recall(rp,xs)
    cr=find_crossover(yb,yp,xs)
    # 小尺度FP变化%
    fp_chg = (fpp[0]-fpb[0])/fpb[0]*100 if fpb[0]>0 else 0
    crossovers.append(cr); fp_changes.append(fp_chg)
    cr_str = f"{cr:.2f}%" if cr else "无"
    print(f"seed{s}: crossover={cr_str}  小FP变化={fp_chg:+.0f}%  (baseFP小={fpb[0]}, p2FP小={fpp[0]})")

cr_valid=[c for c in crossovers if c is not None]
print("\n"+"="*55)
print("Crossover（8种子）")
print("="*55)
print(f"  有效crossover数: {len(cr_valid)}/8")
if cr_valid:
    print(f"  位置: {[f'{c:.2f}' for c in cr_valid]}")
    print(f"  mean={np.mean(cr_valid):.2f}%  std={np.std(cr_valid,ddof=1):.2f}%")
print(f"\nFP变化（8种子）")
print("="*55)
print(f"  {[f'{f:+.0f}' for f in fp_changes]}")
print(f"  mean={np.mean(fp_changes):+.0f}%  std={np.std(fp_changes,ddof=1):.0f}%")

# 存结果
out={'crossovers':[float(c) if c else None for c in crossovers],
     'crossover_mean':float(np.mean(cr_valid)) if cr_valid else None,
     'crossover_std':float(np.std(cr_valid,ddof=1)) if len(cr_valid)>1 else None,
     'fp_changes':[float(f) for f in fp_changes],
     'fp_mean':float(np.mean(fp_changes)),'fp_std':float(np.std(fp_changes,ddof=1))}
with open(f'{BACKUP}/results/crossover_fp_8seeds.json','w') as f:
    json.dump(out,f,indent=2)
print(f"\n已存: {BACKUP}/results/crossover_fp_8seeds.json")
