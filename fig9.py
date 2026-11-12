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
# 3. 录入完整数据 (修复了乱码标签，并对齐 Ours 与 Supervised 统一基准)
# ==========================================
methods = ['Supervised\n(MSE Only)', 'Feature\nMask-GCL', 'Rand Edge\nDrop-GCL',
           'Rand Node\nDrop-GCL', 'Hub-Drop\nGCL', 'Ours\n(Degree-Aware)']

data = {
    'MRCLM':     {'mae': [0.3806, 0.3545, 0.3427, 0.3396, 0.3091, 0.2782], 'rmse': [0.7772, 0.6225, 0.6002, 0.5890, 0.5455, 0.5185]},
    'OMITSJ':    {'mae': [0.2715, 0.2488, 0.2427, 0.2389, 0.2221, 0.1974], 'rmse': [0.6126, 0.5384, 0.4931, 0.4577, 0.4399, 0.3812]},
    'CTLSV':     {'mae': [0.2974, 0.2776, 0.2680, 0.2847, 0.2969, 0.2210], 'rmse': [0.5534, 0.5013, 0.4528, 0.5288, 0.5887, 0.4135]},
    'MRCLASSO':  {'mae': [0.3288, 0.3067, 0.3294, 0.2960, 0.3171, 0.2533], 'rmse': [0.6151, 0.5518, 0.6587, 0.5521, 0.6998, 0.4770]},
    'LRRS':      {'mae': [0.3115, 0.2975, 0.3270, 0.2884, 0.3059, 0.2468], 'rmse': [0.6016, 0.5711, 0.6113, 0.5011, 0.5814, 0.4615]},
    'CITVR':     {'mae': [0.3180, 0.2988, 0.2898, 0.2843, 0.2765, 0.2381], 'rmse': [0.5611, 0.5425, 0.5058, 0.5081, 0.4857, 0.4482]},
    'CITCRC':    {'mae': [0.2681, 0.2490, 0.2647, 0.2480, 0.2563, 0.2032], 'rmse': [0.5361, 0.4798, 0.5000, 0.4280, 0.5090, 0.3975]},
    'CITLA':     {'mae': [0.4533, 0.4387, 0.3543, 0.3561, 0.4109, 0.3341], 'rmse': [0.8513, 0.8113, 0.6856, 0.6836, 0.7945, 0.6380]},
    'CITSO':     {'mae': [0.3207, 0.2906, 0.2817, 0.2864, 0.2739, 0.2382], 'rmse': [0.6008, 0.5723, 0.5375, 0.5130, 0.4940, 0.4445]},
    'CITPI':     {'mae': [0.2674, 0.2446, 0.2388, 0.2275, 0.2179, 0.1965], 'rmse': [0.5538, 0.4923, 0.4623, 0.4353, 0.4191, 0.3882]},
    'CITHSL':    {'mae': [0.2106, 0.2097, 0.1920, 0.1800, 0.1930, 0.1585], 'rmse': [0.4306, 0.4124, 0.3610, 0.3325, 0.3838, 0.3085]}
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
plt.savefig('degree_aware_smart_single_Y.png', dpi=300, bbox_inches='tight')
plt.show()