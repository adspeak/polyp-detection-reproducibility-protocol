# -*- coding: utf-8 -*-
"""
ARCHIVED PRE-AI-ASSISTED VERSION.

Original notebook cell (Google Colab) that generated Figure 4, extracted
verbatim from the authors' working notebook script. The post-AI-assisted
version (`plotting/make_fig4_multiseed_stability.py`) repackages this into a
standalone CLI script (argparse, an optional --results flag to recompute the
bar heights from results/multiseed_8seeds.json instead of the hardcoded
values below, generalised output paths). The plotting logic and the
hardcoded summary values themselves are unchanged; output PDF dimensions
were verified to match the published figure exactly (422.386 x 292.573 pt).
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ==== 期刊规范设置（与原Fig2/3统一）====
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ==== 8种子数据（来自 multiseed_8seeds.json）====
strata = ['Small', 'Medium', 'Large']
baseline_mean = [28.3, 43.2, 47.8]
baseline_std  = [3.1, 5.7, 8.6]
p2_mean       = [29.2, 42.4, 50.3]
p2_std        = [3.5, 4.0, 7.9]

x = np.arange(len(strata))
width = 0.32

fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

bars1 = ax.bar(x - width/2, baseline_mean, width, yerr=baseline_std,
                label='Baseline', color='#d9d9d9', edgecolor='black',
                linewidth=0.8, capsize=4, error_kw={'elinewidth':1.0,'capthick':1.0})
bars2 = ax.bar(x + width/2, p2_mean, width, yerr=p2_std,
                label='Modification (+P2)', color='#4d4d4d', edgecolor='black',
                linewidth=0.8, capsize=4, error_kw={'elinewidth':1.0,'capthick':1.0})

ax.set_ylabel('Recall (%)', fontsize=12)
ax.set_xlabel('Lesion-size stratum', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(strata, fontsize=11)
ax.set_ylim(0, 68)
ax.legend(frameon=False, loc='upper left', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()

# 保存到共享盘
SAVE = '/content/drive/MyDrive/small/LDPolyp_backup/figures'
import os
os.makedirs(SAVE, exist_ok=True)
plt.savefig(f'{SAVE}/fig4_multiseed_stability_8seeds.pdf', bbox_inches='tight')
plt.savefig(f'{SAVE}/fig4_multiseed_stability_8seeds.png', bbox_inches='tight', dpi=300)
plt.show()
print("✅ Fig 4（8种子版）已存共享盘 figures/")
