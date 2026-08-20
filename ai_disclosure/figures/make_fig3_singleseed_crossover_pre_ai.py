# -*- coding: utf-8 -*-
"""
ARCHIVED PRE-AI-ASSISTED VERSION (final "legend font size unified" iteration;
earlier draft iterations exist in the authors' notebook history but are not
archived here).

Original notebook cell (Google Colab) that generated Figure 3, extracted
verbatim from the authors' working notebook script. It relied on
notebook-level context defined in earlier cells:

    OUT        = '/content/ldpolyp'
    DRIVE_PROJ = '/content/drive/MyDrive/LDPolyp_P2_Project'
    import os, glob, numpy as np, matplotlib, matplotlib.pyplot as plt
    from ultralytics import YOLO

The post-AI-assisted version (`plotting/make_fig3_singleseed_crossover.py`)
repackages this into a standalone CLI script (argparse, generalised dataset
and checkpoint paths, English print output). The IoU-matching, sliding-window
recall, and crossover-finding logic is unchanged; its equivalence with the
version adopted in `evaluation/crossover_fp_analysis.py` was verified with
200 randomised trials, and the output PDF dimensions match the published
figure exactly.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob, os
from ultralytics import YOLO

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

DRIVE_PROJ = '/content/drive/MyDrive/LDPolyp_P2_Project'

def box_iou(b,bs):
    if len(bs)==0: return np.array([])
    x1=np.maximum(b[0],bs[:,0]);y1=np.maximum(b[1],bs[:,1]);x2=np.minimum(b[2],bs[:,2]);y2=np.minimum(b[3],bs[:,3])
    iw=np.clip(x2-x1,0,None);ih=np.clip(y2-y1,0,None);inter=iw*ih
    a1=(b[2]-b[0])*(b[3]-b[1]);a2=(bs[:,2]-bs[:,0])*(bs[:,3]-bs[:,1])
    return inter/(a1+a2-inter+1e-9)

def per_gt_hits(weights, imgsz, conf=0.25, iou_m=0.5):
    m=YOLO(weights); recs=[]
    for ip in sorted(glob.glob(f'{OUT}/images/test/*.jpg')):
        lp=f"{OUT}/labels/test/{os.path.basename(ip)[:-4]}.txt"; gts=[]
        if os.path.exists(lp):
            for ln in open(lp):
                p=ln.split()
                if len(p)==5:_,cx,cy,w,h=map(float,p);gts.append([cx,cy,w,h])
        if not gts: continue
        r=m.predict(ip,conf=conf,imgsz=imgsz,verbose=False)[0]
        W,H=r.orig_shape[1],r.orig_shape[0]
        preds=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
        for cx,cy,w,h in gts:
            rel=w*h
            g=[(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]
            io=box_iou(g,preds)
            hit=1 if (len(io)>0 and io.max()>=iou_m) else 0
            recs.append((rel,hit))
    return np.array(recs)

base=per_gt_hits(f'{DRIVE_PROJ}/weights/best_baseline_1280_v2.pt',1280)
p2  =per_gt_hits(f'{DRIVE_PROJ}/weights/best_P2_1280_v2.pt',1280)
print(f"GT总数: baseline={len(base)}  p2={len(p2)}")

def sliding_recall(recs, xs, win=0.5):
    rel=recs[:,0]*100; hit=recs[:,1]
    ys=[]
    for x in xs:
        lo,hi=x/(1+win),x*(1+win)
        mask=(rel>=lo)&(rel<hi)
        ys.append(hit[mask].mean()*100 if mask.sum()>=10 else np.nan)
    return np.array(ys)

xs=np.logspace(np.log10(0.3),np.log10(30),40)
yb=sliding_recall(base,xs); yp=sliding_recall(p2,xs)

diff=yp-yb; cross=None
valid=~np.isnan(diff)
xv=xs[valid]; dv=diff[valid]
for i in range(len(dv)-1):
    if dv[i]>0 and dv[i+1]<=0:
        x0,x1=xv[i],xv[i+1]; d0,d1=dv[i],dv[i+1]
        cross=x0+(x1-x0)*d0/(d0-d1)
        break
print(f"\n★ 交叉点: {cross:.2f}%" if cross else "未找到")

# ==== 画图（修正：统一图例字号12、marker/线宽与Fig1/2/4一致规范） ====
fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

ax.plot(xs, yb, 'o-', color='#999999', lw=1.8, ms=5,
        markerfacecolor='#d9d9d9', markeredgecolor='#333333', markeredgewidth=0.7,
        label='Baseline')
ax.plot(xs, yp, 's--', color='#333333', lw=1.8, ms=5,
        markerfacecolor='#4d4d4d', markeredgecolor='#000000', markeredgewidth=0.7,
        label='Modification (+P2)')

if cross:
    ax.axvline(cross, ls=':', color='#666666', lw=1.2, alpha=0.9)
    ax.text(cross, 6, f'crossover\n{cross:.2f}%', color='#333333',
            ha='center', fontsize=9.5)

ax.set_xscale('log')
ax.set_xlabel('Polyp relative area (%, log scale)', fontsize=12)
ax.set_ylabel('Local recall (%)', fontsize=12)
ax.set_ylim(0, 100)
# 关键修正：字号从10统一改为12（与Fig1/2/4一致），frameon保持一致
ax.legend(frameon=False, loc='upper left', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(f'{DRIVE_PROJ}/figures/fig3_singleseed_crossover.pdf', bbox_inches='tight')
plt.savefig(f'{DRIVE_PROJ}/figures/fig3_singleseed_crossover.png', bbox_inches='tight', dpi=300)
plt.show()
print("\n✅ Fig 3（图例字号统一版）已存云盘")
