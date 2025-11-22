# import numpy as np
# import matplotlib.pyplot as plt

# def plot_multiple_loss_curves_by_epoch(train_losses_list, valid_losses_list, model_names, n_epochs):
#     """根据模型数量自适应颜色、线型和 marker 绘制 loss 曲线。"""

#     plt.figure(figsize=(14, 8))

#     # 颜色：使用 tab20 colormap，适合多模型
#     colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

#     # 更多线型（循环不重复）
#     line_styles = ['-', '--', '-.', ':']

#     # 多种 marker
#     markers = ['o', 's', '^', 'v', '>', '<', 'd', 'p', '*', 'h', 'X']

#     epochs = np.arange(1, n_epochs + 1)

#     for i, (train_losses, valid_losses, model_name) in enumerate(
#         zip(train_losses_list, valid_losses_list, model_names)
#     ):
#         color = colors[i % len(colors)]
#         ls = line_styles[i % len(line_styles)]
#         mk_train = markers[(2*i) % len(markers)]
#         mk_valid = markers[(2*i+1) % len(markers)]

#         # ---- Train loss ----
#         plt.plot(
#             epochs, train_losses,
#             label=f"{model_name} - Train",
#             color=color,
#             linestyle=ls,
#             linewidth=2,
#             marker=mk_train,
#             markersize=4
#         )

#         # ---- Valid loss ----
#         plt.plot(
#             epochs, valid_losses,
#             label=f"{model_name} - Valid",
#             color=color,
#             linestyle=ls,
#             linewidth=2,
#             marker=mk_valid,
#             markersize=4,
#             alpha=0.8
#         )

#         # ---- Mark minimum valid loss ----
#         min_valid_loss = min(valid_losses)
#         min_epoch = valid_losses.index(min_valid_loss) + 1

#         plt.scatter(min_epoch, min_valid_loss, color=color, s=100, marker='*', zorder=5)
#         plt.annotate(
#             f'{model_name}: {min_valid_loss:.4f}',
#             xy=(min_epoch, min_valid_loss),
#             xytext=(10, 8),
#             textcoords='offset points',
#             fontsize=9,
#             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.25),
#             arrowprops=dict(arrowstyle='->', color=color)
#         )

#     plt.title("Multiple Models - Training and Validation Loss", fontsize=16)
#     plt.xlabel("Epochs", fontsize=14)
#     plt.ylabel("Loss", fontsize=14)

#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=10, loc="upper right", ncol=2)

#     plt.xticks(np.arange(1, n_epochs + 1, max(1, n_epochs // 10)))

#     plt.tight_layout()
#     plt.savefig(f'multiple_models_loss_curves_by_epoch_{n_epochs}.png', dpi=300)
#     plt.show()

#     # -------- Summary -------
#     print("\n=== 多模型训练摘要（基于Epoch） ===")
#     for train_losses, valid_losses, model_name in zip(train_losses_list, valid_losses_list, model_names):
#         min_valid_loss = min(valid_losses)
#         min_valid_epoch = valid_losses.index(min_valid_loss) + 1
#         min_train_loss = min(train_losses)
#         min_train_epoch = train_losses.index(min_train_loss) + 1

#         print(f"\n{model_name}:")
#         print(f"  最终训练损失: {train_losses[-1]:.4f}")
#         print(f"  最终验证损失: {valid_losses[-1]:.4f}")
#         print(f"  最小训练损失: {min_train_loss:.4f} (Epoch {min_train_epoch})")
#         print(f"  最小验证损失: {min_valid_loss:.4f} (Epoch {min_valid_epoch})")
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_metrics_by_epoch(
    train_losses_list, valid_losses_list,
    train_acc_list, valid_acc_list,
    model_names, n_epochs
):
    """绘制多个模型的 Loss & Accuracy 曲线，横轴为 epoch"""
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    line_styles = ['-', '--', '-.', ':']
    epochs = list(range(1, n_epochs + 1))

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # ==========================
    # 1) LOSS CURVES
    # ==========================
    ax = axes[0]
    for i, (train_losses, valid_losses, name) in enumerate(zip(
        train_losses_list, valid_losses_list, model_names
    )):
        color = colors[i]
        ls_train = line_styles[i % len(line_styles)]
        ls_valid = line_styles[(i+1) % len(line_styles)]

        # train loss
        ax.plot(epochs, train_losses, 
                color=color, linestyle=ls_train, linewidth=2,
                label=f"{name} - Train Loss")

        # valid loss
        ax.plot(epochs, valid_losses, 
                color=color, linestyle=ls_valid, linewidth=2,
                label=f"{name} - Valid Loss")

        # 最小 val loss 标记
        min_v = min(valid_losses)
        min_ep = valid_losses.index(min_v) + 1
        ax.scatter(min_ep, min_v, color=color, s=80, marker="*")
        ax.annotate(f"{name}: {min_v:.4f}",
                    xy=(min_ep, min_v),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle="->", color=color)
                   )

    ax.set_title("Training & Validation Loss", fontsize=16)
    ax.set_ylabel("Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)

    # ==========================
    # 2) ACCURACY CURVES
    # ==========================
    ax = axes[1]
    for i, (train_acc, valid_acc, name) in enumerate(zip(
        train_acc_list, valid_acc_list, model_names
    )):
        color = colors[i]
        ls_train = line_styles[i % len(line_styles)]
        ls_valid = line_styles[(i+1) % len(line_styles)]

        # train acc
        ax.plot(epochs, train_acc,
                color=color, linestyle=ls_train, linewidth=2,
                label=f"{name} - Train Acc")

        # valid acc
        ax.plot(epochs, valid_acc,
                color=color, linestyle=ls_valid, linewidth=2,
                label=f"{name} - Valid Acc")

        # 最大 val acc 标记
        max_v = max(valid_acc)
        max_ep = valid_acc.index(max_v) + 1
        ax.scatter(max_ep, max_v, color=color, s=80, marker="o")
        ax.annotate(f"{name}: {max_v:.2f}",
                    xy=(max_ep, max_v),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle="->", color=color)
                   )

    ax.set_title("Training & Validation Accuracy", fontsize=16)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)

    plt.xticks(np.arange(1, n_epochs + 1, max(1, n_epochs // 10)))
    plt.tight_layout()
    plt.savefig(f"multiple_models_loss_acc_curves_{n_epochs}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

    # ================
    # Summary Print
    # ================
    print("\n=== Training Summary ===")
    for name, tl, vl, ta, va in zip(
        model_names, train_losses_list, valid_losses_list,
        train_acc_list, valid_acc_list
    ):
        print(f"\n📌 {name}")
        print(f"  Final loss:      train={tl[-1]:.4f}, valid={vl[-1]:.4f}")
        print(f"  Min val loss:    {min(vl):.4f}  (epoch {vl.index(min(vl))+1})")
        print(f"  Final accuracy:  train={ta[-1]:.3f}, valid={va[-1]:.3f}")
        print(f"  Max val acc:     {max(va):.3f}  (epoch {va.index(max(va))+1})")
