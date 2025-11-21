import numpy as np
import matplotlib.pyplot as plt
def plot_multiple_loss_curves_by_epoch(train_losses_list, valid_losses_list, model_names, n_epochs):
    """绘制多个模型的训练和验证损失曲线，横轴为epoch"""
    plt.figure(figsize=(14, 8))
    
    # 定义颜色和线型
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    line_styles = ['-', '--', '-.', ':']
    
    # 创建epoch列表
    epochs = list(range(1, n_epochs + 1))
    
    # 绘制每个模型的曲线
    for i, (train_losses, valid_losses, model_name) in enumerate(zip(
        train_losses_list, valid_losses_list, model_names)):
        
        color = colors[i]
        line_style = line_styles[i % len(line_styles)]
        
        # 绘制训练损失（每个epoch）
        plt.plot(epochs, train_losses, 
                color=color, linestyle=line_style, linewidth=2,
                label=f'{model_name} - Train', marker='s', markersize=4)
        
        # 绘制验证损失（每个epoch）
        plt.plot(epochs, valid_losses, 
                color=color, linestyle=line_style, linewidth=2,
                label=f'{model_name} - Valid', marker='o', markersize=4)
        
        # 标记每个模型的最小验证损失
        min_valid_loss = min(valid_losses)
        min_epoch = valid_losses.index(min_valid_loss) + 1  # +1因为索引从0开始
        plt.scatter(min_epoch, min_valid_loss, color=color, s=100, marker='*', zorder=5)
        plt.annotate(f'{model_name}: {min_valid_loss:.4f}', 
                    xy=(min_epoch, min_valid_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color))
    
    plt.title(f'Multiple Models - Training and Validation Loss (Epochs)', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=10, loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度为每1个epoch
    plt.xticks(np.arange(1, n_epochs + 1, max(1, n_epochs // 10)))
    
    plt.tight_layout()
    plt.savefig(f'multiple_models_loss_curves_by_epoch_{n_epochs}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印所有模型的训练摘要
    print(f"\n=== 多模型训练摘要（基于Epoch）===")
    for i, (train_losses, valid_losses, model_name) in enumerate(zip(
        train_losses_list, valid_losses_list, model_names)):
        
        min_valid_loss = min(valid_losses)
        min_epoch = valid_losses.index(min_valid_loss) + 1
        
        # 计算训练损失的最小值
        min_train_loss = min(train_losses)
        min_train_epoch = train_losses.index(min_train_loss) + 1
        
        print(f"\n{model_name}:")
        print(f"  最终训练损失: {train_losses[-1]:.4f}")
        print(f"  最终验证损失: {valid_losses[-1]:.4f}")
        print(f"  最低训练损失: {min_train_loss:.4f} (在第 {min_train_epoch} 个epoch)")
        print(f"  最低验证损失: {min_valid_loss:.4f} (在第 {min_epoch} 个epoch)")