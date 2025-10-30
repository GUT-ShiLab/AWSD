import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def barchart(save_to_png = False):
    # 读取 Excel 数据
    df = pd.read_excel('C:\\Users\\l\\Desktop\\code\\论文\\实验结果\\protect.xlsx')  # 替换为实际 Excel 文件名

    # 提取数据
    classes = df['class'].unique()
    f1_scores = df[df['Type'] == 'f1']['score']
    auc_scores = df[df['Type'] == 'auc']['score']

    # 设置画布
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(classes))  # 类别位置

    # 绘制柱子
    plt.bar(x - bar_width/2, f1_scores, width=bar_width, label='f1', color='blue')
    plt.bar(x + bar_width/2, auc_scores, width=bar_width, label='auc', color='orange')

    # 设置坐标轴标签和标题
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Grouped Bar Chart of Scores', fontsize=14)

    # 设置 x 轴刻度标签
    plt.xticks(x, classes, rotation=45, ha='right')

    # 添加图例
    plt.legend()

    # 调整布局
    plt.tight_layout()

    if save_to_png:
        plt.savefig('grouped_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()


# barchart(save_to_png = True)
import torch
import pdb
result = torch.load(f'plots/average F1 plots/plots for _HierarchicalDistillation_____infant_None__CNNLSTMTransformer_padded_None.pth',weights_only=False)

result['mean']['validation auc']
pdb.set_trace()

