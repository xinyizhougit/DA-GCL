# ==========================================
# 1. 修复 PyCharm/IntelliJ 环境兼容性报错
# ==========================================
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ==========================================
# 2. 全局字号与样式设置
# ==========================================
FONT_TITLE = 14       # 子图标题字号
FONT_LABEL = 10       # X/Y 轴刻度标签字号
FONT_TEXT_BAR = 8     # 柱状图上的数值字号
FONT_TEXT_LINE = 8    # 折线图上的数值字号
FONT_LEGEND = 8       # 图例字号

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==========================================
# 3. 录入完整数据 (修复了乱码标签，并对齐 Ours 与 Physical 统一基准)
# ==========================================
methods = ['Physical\nOnly', 'Semantic\nOnly', 'Static\nPearson',
           'Predefined\nFusion', 'GAT-style\nMasking', 'Ours\n(Adaptive)']

data = {
    'MRCLM':     {'mae': [0.3736, 0.3622, 0.3462, 0.3031, 0.2829, 0.2782], 'rmse': [0.7192, 0.6794, 0.6389, 0.6178, 0.5589, 0.5185]},
    'OMITSJ':    {'mae': [0.2724, 0.2580, 0.2468, 0.2211, 0.2041, 0.1974], 'rmse': [0.5173, 0.4879, 0.4844, 0.4225, 0.4273, 0.3812]},
    'CTLSV':     {'mae': [0.3500, 0.3171, 0.2968, 0.2573, 0.2410, 0.2210], 'rmse': [0.5599, 0.5375, 0.5117, 0.4822, 0.4550, 0.4135]},
    'MRCLASSO':  {'mae': [0.4435, 0.3165, 0.2749, 0.2779, 0.2724, 0.2533], 'rmse': [0.6448, 0.5581, 0.5910, 0.5366, 0.4790, 0.4770]},
    'LRRS':      {'mae': [0.3392, 0.3113, 0.2992, 0.2771, 0.2713, 0.2468], 'rmse': [0.6451, 0.5870, 0.5742, 0.5115, 0.5242, 0.4615]},
    'CITVR':     {'mae': [0.3145, 0.3473, 0.2892, 0.2822, 0.2718, 0.2381], 'rmse': [0.5445, 0.5894, 0.5494, 0.5387, 0.5577, 0.4482]},
    'CITCRC':    {'mae': [0.3516, 0.2568, 0.2493, 0.2589, 0.2371, 0.2032], 'rmse': [0.5404, 0.5380, 0.4742, 0.4682, 0.4875, 0.3975]},
    'CITLA':     {'mae': [0.4677, 0.3713, 0.3599, 0.3216, 0.3536, 0.3341], 'rmse': [0.9015, 0.6756, 0.7586, 0.7716, 0.6774, 0.6380]},
    'CITSO':     {'mae': [0.3164, 0.3105, 0.2872, 0.2714, 0.2729, 0.2382], 'rmse': [0.5900, 0.5986, 0.5792, 0.5516, 0.5309, 0.4445]},
    'CITPI':     {'mae': [0.3253, 0.2343, 0.3273, 0.2814, 0.2865, 0.1965], 'rmse': [0.4574, 0.4272, 0.4567, 0.4257, 0.4304, 0.3882]},
    'CITHSL':    {'mae': [0.2739, 0.2237, 0.2137, 0.1684, 0.1604, 0.1585], 'rmse': [0.4759, 0.4877, 0.4577, 0.3301, 0.3701, 0.3085]}
}

# ==========================================
# 4. 智能判断与绘图核心
# ==========================================
fig = plt.figure(figsize=(18, 12))
gs_outer = GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.55)

x = np.arange(len(methods))
width = 0.45

color_bar = '#1f77b4'
color_line = '#003366'

# 智能断层阈值 (RMSE最小值 与 MAE最大值 之间的空白落差大于此值则截断)
GAP_THRESHOLD = 0.15

for idx, (region, metrics) in enumerate(data.items()):
    row = idx // 4
    col = idx % 4
    cell = gs_outer[row, col]

    mae_vals = metrics['mae']
    rmse_vals = metrics['rmse']

    max_mae = max(mae_vals)
    min_rmse = min(rmse_vals)
    max_rmse = max(rmse_vals)

    gap = min_rmse - max_mae

    # === 情况 A: 落差较大，启用真实的单 Y 轴截断 ===
    if gap > GAP_THRESHOLD:
        bot_ylim = [0, max_mae + 0.05]
        top_ylim = [min_rmse - 0.05, max_rmse + 0.1]

        ratio_top = top_ylim[1] - top_ylim[0]
        ratio_bot = bot_ylim[1] - bot_ylim[0]

        gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, hspace=0.08, height_ratios=[ratio_top, ratio_bot])
        ax_top = fig.add_subplot(gs_inner[0, 0])
        ax_bot = fig.add_subplot(gs_inner[1, 0])

        for ax in [ax_top, ax_bot]:
            ax.bar(x, mae_vals, width, label='MAE', color=color_bar, zorder=3)
            ax.plot(x, rmse_vals, label='RMSE', color=color_line, marker='o', markersize=5, linewidth=1.5, zorder=4)
            ax.tick_params(axis='y', labelsize=FONT_LABEL)
            ax.spines['right'].set_visible(False)

        ax_top.set_ylim(top_ylim)
        ax_bot.set_ylim(bot_ylim)

        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(methods, rotation=35, ha='right', fontsize=FONT_LABEL)

        # 画截断斜线 (//)
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        # 添加标签
        for i, val in enumerate(mae_vals):
            ax_bot.annotate(f'{val:.4f}', xy=(x[i], val), xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=FONT_TEXT_BAR)
        for i, val in enumerate(rmse_vals):
            ax_top.annotate(f'{val:.4f}', xy=(x[i], val), xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=FONT_TEXT_LINE)

        ax_top.set_title(region, loc='left', fontsize=FONT_TITLE, pad=10)
        ax_top.legend(loc='upper right', fontsize=FONT_LEGEND)

    # === 情况 B: 落差较小或重叠，启用连续的不截断单 Y 轴 ===
    else:
        ax = fig.add_subplot(cell)

        ax.bar(x, mae_vals, width, label='MAE', color=color_bar, zorder=3)
        ax.plot(x, rmse_vals, label='RMSE', color=color_line, marker='o', markersize=5, linewidth=1.5, zorder=4)

        ax.set_ylim(0, max_rmse * 1.15)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=FONT_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_LABEL)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for i, val in enumerate(mae_vals):
            ax.annotate(f'{val:.4f}', xy=(x[i], val), xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=FONT_TEXT_BAR)
        for i, val in enumerate(rmse_vals):
            ax.annotate(f'{val:.4f}', xy=(x[i], val), xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONT_TEXT_LINE)

        ax.set_title(region, loc='left', fontsize=FONT_TITLE, pad=10)
        ax.legend(loc='upper right', fontsize=FONT_LEGEND)

# 隐藏多余的子图格子
if len(fig.axes) > 11:
    gs_outer[2, 3]

ax_empty = fig.add_subplot(gs_outer[2, 3])
ax_empty.axis('off')

plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
plt.savefig('dual_view_smart_single_Y.png', dpi=300, bbox_inches='tight')
plt.show()