import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs('./imgs', exist_ok=True)


def normalize_head_importance(head_importance):
    layer_keys = sorted(head_importance.keys(), key=lambda x: int(x.split("_")[1]))
    num_layers = len(layer_keys)
    num_heads = len(head_importance[layer_keys[0]])

    attn_matrix = np.zeros((num_layers, num_heads))

    for i, key in enumerate(layer_keys):
        attn_tensor = head_importance[key]  # shape: (heads,)
        attn_matrix[i] = attn_tensor.detach().cpu().numpy()  # 移动到 CPU 并转 numpy
    
    return attn_matrix

def visualize_head_importance(head_importance, task_id):
    head_importance = normalize_head_importance(head_importance)
    np.save(f"./imgs/head_importance_task_{task_id}.npy", head_importance)
    # 设置图形大小
    plt.figure(figsize=(12, 10))
    # 绘制热力图
    ax = sns.heatmap(
        head_importance,
        cmap="Oranges",
        cbar_kws={'label': 'Weight'},
        linewidths=0.1,
        xticklabels=1,  # 开启横坐标标签
        yticklabels=1   # 开启纵坐标标签
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Weight", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # 设置轴标签
    plt.xlabel("Head", fontsize=14)
    plt.ylabel("Layer", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 显示图像
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f"./imgs/head_importance_heatmap_task_{task_id}.pdf", dpi=300, bbox_inches='tight')
    plt.close()