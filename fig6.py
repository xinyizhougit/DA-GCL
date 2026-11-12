# ==========================================
# 1. 修复 PyCharm/IntelliJ 环境兼容性报错
# ==========================================
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ==========================================
# 2. 全局字号与样式设置 (按照您的要求)
# ==========================================
FONT_TITLE = 14       # 子图标题字号
FONT_LABEL = 10       # X/Y 轴刻度标签字号
FONT_TEXT_BAR = 8     # 柱状图上的数值字号
FONT_TEXT_LINE = 8    # 折线图上的数值字号
FONT_LEGEND = 8       # 图例字号

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==========================================
# 3. 录入完整数据
# ==========================================
methods = ['Raw\nData', 'Linear\nInterp.', 'Matrix\nFactor.',
           'Euclidean\nKNN', 'DTW-KNN\n(w/o GMM)', 'Ours\n(+GMM)']

data = {
    'MRCLM':     {'mae': [0.3950, 0.3662, 0.3512, 0.3234, 0.3076, 0.2782], 'rmse': [0.7501, 0.6864, 0.6686, 0.6127, 0.5792, 0.5185]},
    'OMITSJ':    {'mae': [0.2764, 0.2551, 0.2521, 0.2218, 0.2205, 0.1974], 'rmse': [0.5199, 0.4845, 0.4781, 0.4265, 0.4230, 0.3812]},
    'CTLSV':     {'mae': [0.3094, 0.2892, 0.2868, 0.2541, 0.2488, 0.2210], 'rmse': [0.5960, 0.5524, 0.5379, 0.4729, 0.4613, 0.4135]},
    'MRCLASSO':  {'mae': [0.3597, 0.3386, 0.3165, 0.2961, 0.2784, 0.2533], 'rmse': [0.6885, 0.6413, 0.6040, 0.5574, 0.5355, 0.4770]},
    'LRRS':      {'mae': [0.3392, 0.3300, 0.3261, 0.2989, 0.2873, 0.2468], 'rmse': [0.6432, 0.6308, 0.6200, 0.5603, 0.5422, 0.4615]},
    'CITVR':     {'mae': [0.3333, 0.3122, 0.3092, 0.2723, 0.2629, 0.2381], 'rmse': [0.6244, 0.5954, 0.5904, 0.5194, 0.5049, 0.4482]},
    'CITCRC':    {'mae': [0.2832, 0.2733, 0.2724, 0.2582, 0.2461, 0.2032], 'rmse': [0.5513, 0.5286, 0.5235, 0.5056, 0.4884, 0.3975]},
    'CITLA':     {'mae': [0.4945, 0.3893, 0.3813, 0.3632, 0.3572, 0.3341], 'rmse': [0.9522, 0.7511, 0.7317, 0.7020, 0.6932, 0.6380]},
    'CITSO':     {'mae': [0.3525, 0.2816, 0.2646, 0.2644, 0.2431, 0.2382], 'rmse': [0.6770, 0.5314, 0.5136, 0.5086, 0.4501, 0.4445]},
    'CITPI':     {'mae': [0.2751, 0.2580, 0.2523, 0.2218, 0.2148, 0.1965], 'rmse': [0.5541, 0.5174, 0.5042, 0.4460, 0.4361, 0.3882]},
    'CITHSL':    {'mae': [0.2187, 0.2107, 0.2101, 0.1924, 0.1859, 0.1585], 'rmse': [0.4281, 0.4235, 0.4186, 0.3797, 0.3709, 0.3085]}
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

        # 严格按照数值跨度计算物理高度，保证“单Y轴”视觉上的绝对公平
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

        # MAE和RMSE画在同一个绝对坐标系下
        ax.bar(x, mae_vals, width, label='MAE', color=color_bar, zorder=3)
        ax.plot(x, rmse_vals, label='RMSE', color=color_line, marker='o', markersize=5, linewidth=1.5, zorder=4)

        # Y轴从 0 连续到最大值
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
fig.axes[-1].axis('off') if len(fig.axes) == 12 else None # 容错处理，取决于拆分数量
if idx == 10:
    gs_outer[2, 3] # 清理空白格

plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
plt.savefig('preprocessing_smart_single_Y.png', dpi=300, bbox_inches='tight')
plt.show()