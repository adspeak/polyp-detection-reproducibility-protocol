# -*- coding: utf-8 -*-
"""
ARCHIVED PRE-AI-ASSISTED VERSION (final "corrected" iteration; earlier
draft iterations exist in the authors' notebook history but are not
archived here).

Original notebook cell (Google Colab) that generated Figure 2, extracted
verbatim from the authors' working notebook script. It relied on
notebook-level context defined in earlier cells:

    DRIVE_PROJ = '/content/drive/MyDrive/LDPolyp_P2_Project'
    import os, glob, numpy as np, matplotlib, matplotlib.pyplot as plt, shutil

The post-AI-assisted version (`plotting/make_fig2_dataset_distribution.py`)
repackages this into a standalone CLI script (argparse, generalised label
directory paths, English print output). The plotting logic -- including the
two-step vertical-space fix noted in the original comments below -- is
unchanged; output PDF dimensions were verified to match the published figure
exactly (422.386 x 292.969 pt).
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, glob, shutil

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

DRIVE_PROJ = '/content/drive/MyDrive/LDPolyp_P2_Project'

if not os.path.exists('/content/ldpolyp'):
    shutil.unpack_archive(f'{DRIVE_PROJ}/ldpolyp_dataset.zip', '/content/ldpolyp')
if not os.path.exists('/content/kvasir'):
    shutil.copytree('/content/drive/MyDrive/GI_Project/datasets/kvasir_seg_yolo', '/content/kvasir')

def get_areas(label_dir):
    areas = []
    for lp in glob.glob(f'{label_dir}/*.txt'):
        for ln in open(lp):
            p = ln.split()
            if len(p) == 5:
                _, cx, cy, w, h = map(float, p)
                areas.append(w * h * 100)
    return np.array(areas)

ld_areas = get_areas('/content/ldpolyp/labels/train')
kv_areas = np.concatenate([
    get_areas('/content/kvasir/labels/train'),
    get_areas('/content/kvasir/labels/val')
])

print(f"LDPolypVideo: {len(ld_areas)} 个框, 中位数={np.median(ld_areas):.2f}%")
print(f"Kvasir-SEG:   {len(kv_areas)} 个框, 中位数={np.median(kv_areas):.2f}%")

# ==== 画图（修正：留出顶部空间 + 文字位置下移） ====
fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

bins = np.logspace(np.log10(0.1), np.log10(100), 35)

ax.hist(ld_areas, bins=bins, alpha=0.75, color='#4d4d4d', edgecolor='black',
        linewidth=0.5, label='LDPolypVideo', density=True)
ax.hist(kv_areas, bins=bins, alpha=0.55, color='#d9d9d9', edgecolor='black',
        linewidth=0.5, label='Kvasir-SEG', density=True)

clinical_ref = 2.04
ax.axvline(clinical_ref, ls=':', color='#333333', lw=1.3)

# 关键修正1：先扩大y轴上限，留出顶部空间
ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.18)

# 关键修正2：文字位置系数从0.92改为0.80（相对新的更高的ymax）
ax.text(clinical_ref, ax.get_ylim()[1]*0.80, '5 mm reference\n(2.04%)',
        color='#333333', fontsize=8.5, ha='center')

ax.set_xscale('log')
ax.set_xlabel('Relative bounding-box area (%, log scale)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(frameon=False, loc='upper right', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(f'{DRIVE_PROJ}/figures/fig2_dataset_distribution.pdf', bbox_inches='tight')
plt.savefig(f'{DRIVE_PROJ}/figures/fig2_dataset_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
print("\n✅ Fig 2（修正版）已存云盘")
