import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
import torch
from torch.utils.data import DataLoader

def plot_tsne_features_2d(X, y, model_name):
    """绘制t-SNE特征可视化图"""
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric='euclidean',
        n_jobs=8,
        random_state=42,
        verbose=True
    )
    features_tsne = tsne.fit(X) 
    plt.figure(figsize=(12, 10))
    
    # 获取唯一标签和颜色
    # np.unique()去除重复;结果按升序排列;返回一个 numpy.ndarray
    unique_labels = np.unique(y)
    selected_labels=np.random.choice(unique_labels, size=10, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_labels)))
    
    # 为每个类别绘制散点
    for i, label in enumerate(selected_labels):
        mask = y == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                   color=colors[i], label=f'Class {label}', 
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    # plt.title(f't-SNE Visualization of {model_name} Features\n({n_samples} samples)', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.figtext(0.02, 0.02, 
                f'Total samples: {len(y)}\nUnique classes: {len(unique_labels)}\nFeature dim: {X.shape[1]}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_tsne_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"t-SNE可视化完成! 图像保存为: {model_name}_tsne_features_2d.png")
def plot_tsne_features_3d(X,y,model_name):
    """
    特征t-SNE可视化
    Args:
        max_samples: 最大样本数（避免内存问题）
    """
    # 确保X和y是numpy数组
    X = np.asarray(X)
    y = np.asarray(y)

    # 使用t-SNE降维
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    features_tsne = tsne.fit(X)
    # 3D可视化
    fig = plt.figure(figsize=(14, 12))  # 创建画布
    ax = fig.add_subplot(111, projection='3d')  # 添加3D坐标轴

    unique_labels = np.unique(y)
    selected_labels=np.random.choice(unique_labels, size=10, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_labels)))

    for i, label in enumerate(selected_labels):
        # 筛选出当前数字i的所有点，用对应的颜色绘制
        mask = y == label
        ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                    color=colors[i], label=f'Class {label}', 
                    alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    # 设置坐标轴标签
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D t-SNE Visualization of Handwritten Digits')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图表外侧
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形
    plt.savefig(f'{model_name}_tsne_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"t-SNE可视化完成! 图像保存为: {model_name}_tsne_features_3d.png")
