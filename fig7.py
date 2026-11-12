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
# 3. 录入完整数据 (精确提取自 Spectral Denoise 原图)
# ==========================================
methods = ['Raw\nTopology', 'DropEdge', 'Trunc.\nSVD', 'Static\nGFT', 'Ours\n(EFT)']

data = {
    'MRCLM':     {'mae': [0.3554, 0.3435, 0.3208, 0.3004, 0.2782], 'rmse': [0.6750, 0.6513, 0.6219, 0.5655, 0.5185]},
    'OMITSJ':    {'mae': [0.2555, 0.2404, 0.2282, 0.2143, 0.1974], 'rmse': [0.4905, 0.4798, 0.4449, 0.4173, 0.3812]},
    'CTLSV':     {'mae': [0.2841, 0.2669, 0.2582, 0.2356, 0.2210], 'rmse': [0.5389, 0.5203, 0.4826, 0.4455, 0.4135]},
    'MRCLASSO':  {'mae': [0.3217, 0.3122, 0.2937, 0.2751, 0.2533], 'rmse': [0.6236, 0.5976, 0.5597, 0.5197, 0.4770]},
    'LRRS':      {'mae': [0.3151, 0.3024, 0.2823, 0.2669, 0.2468], 'rmse': [0.6081, 0.5827, 0.5478, 0.5096, 0.4615]},
    'CITVR':     {'mae': [0.3067, 0.2918, 0.2796, 0.2548, 0.2381], 'rmse': [0.5836, 0.5640, 0.5202, 0.4923, 0.4482]},
    'CITCRC':    {'mae': [0.2663, 0.2488, 0.2393, 0.2222, 0.2032], 'rmse': [0.5201, 0.4937, 0.4711, 0.4333, 0.3975]},
    'CITLA':     {'mae': [0.4318, 0.4085, 0.3838, 0.3619, 0.3341], 'rmse': [0.8314, 0.7994, 0.7464, 0.7013, 0.6380]},
    'CITSO':     {'mae': [0.3030, 0.2911, 0.2735, 0.2591, 0.2382], 'rmse': [0.5791, 0.5637, 0.5301, 0.4907, 0.4445]},
    'CITPI':     {'mae': [0.2648, 0.2423, 0.2240, 0.2104, 0.1965], 'rmse': [0.5072, 0.4861, 0.4537, 0.4219, 0.3882]},
    'CITHSL':    {'mae': [0.2021, 0.1961, 0.1831, 0.1723, 0.1585], 'rmse': [0.4013, 0.3826, 0.3574, 0.3330, 0.3085]}
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

# 隐藏第12个多余的子图
if len(fig.axes) > 11:
    for i in range(len(fig.axes)):
        # 确保最后一个空的格子被隐藏
        if i >= len(data.items()) * 2: # 取决于是否有截断，简单起见直接清理 [2, 3] 位置
            pass
gs_outer[2, 3]

# 额外清理右下角的空白区域
ax_empty = fig.add_subplot(gs_outer[2, 3])
ax_empty.axis('off')

plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
plt.savefig('spectral_denoise_smart_single_Y.png', dpi=300, bbox_inches='tight')
plt.show()