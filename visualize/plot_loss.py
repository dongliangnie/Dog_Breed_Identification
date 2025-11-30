import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
def plot_multiple_metrics_by_epoch(
    train_losses_list, valid_losses_list,
    train_acc_list, valid_acc_list,
    model_names, n_epochs,
    save_prefix="metrics"
):
    """
    绘制两张图（每张 1x2 横向子图）：
      Figure A (Train): [ Train Loss | Train Acc ]
      Figure B (Valid): [ Valid Loss | Valid Acc ]
    参数：
      - lists: 每个都是 list of lists，长度 = 模型数量
      - model_names: list of names
      - n_epochs: int
      - save_prefix: 文件名前缀
    """
    assert len(train_losses_list) == len(valid_losses_list) == len(train_acc_list) == len(valid_acc_list) == len(model_names), \
        "输入列表长度必须一致"

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    line_styles = ['-', '--', '-.', ':']
    epochs = np.arange(1, n_epochs + 1)

    def _place_ann(ax, x, y, text, color, idx, used_y):
        # 简单左右交替 + 不严格冲突避让
        dx = 30 if idx % 2 == 0 else -60
        dy = 10
        # 避免高度冲突（简单版）
        while any(abs((y + dy*0.01) - yy) < 0.02 for yy in used_y):
            dy += 12
        used_y.append(y + dy*0.01)
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, bbox=dict(facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle='->', color=color))

    # -----------------------
    # Figure 1: TRAIN (1x2)
    # -----------------------
    fig, (ax_loss_t, ax_acc_t) = plt.subplots(1, 2, figsize=(20, 10), sharex=True)

    used_y_loss = []
    used_y_acc  = []
    for i, name in enumerate(model_names):
        color = colors[i]
        ls = line_styles[i % len(line_styles)]

        # train loss
        tl = train_losses_list[i]
        ax_loss_t.plot(epochs, tl, color=color, linestyle=ls, linewidth=2, label=name)
        min_v = min(tl); min_ep = tl.index(min_v) + 1
        ax_loss_t.scatter(min_ep, min_v, color=color, s=60, marker='*')
        _place_ann(ax_loss_t, min_ep, min_v, f"{name} {min_v:.4f}", color, i, used_y_loss)

        # train acc
        ta = train_acc_list[i]
        ax_acc_t.plot(epochs, ta, color=color, linestyle=ls, linewidth=2, label=name)
        max_v = max(ta); max_ep = ta.index(max_v) + 1
        ax_acc_t.scatter(max_ep, max_v, color=color, s=60, marker='o')
        _place_ann(ax_acc_t, max_ep, max_v, f"{name} {max_v:.3f}", color, i, used_y_acc)

    ax_loss_t.set_title("Train Loss", fontsize=14)
    ax_loss_t.set_xlabel("Epoch")
    ax_loss_t.set_ylabel("Loss")
    ax_loss_t.grid(alpha=0.25)
    ax_loss_t.legend(fontsize=9, ncol=1)

    ax_acc_t.set_title("Train Accuracy", fontsize=14)
    ax_acc_t.set_xlabel("Epoch")
    ax_acc_t.set_ylabel("Accuracy")
    ax_acc_t.grid(alpha=0.25)
    ax_acc_t.legend(fontsize=9, ncol=1)

    # plt.suptitle("Training Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"./result/{save_prefix}_train_{n_epochs}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ------------------------
    # Figure 2: VALID (1x2)
    # ------------------------
    fig, (ax_loss_v, ax_acc_v) = plt.subplots(1, 2, figsize=(20, 10), sharex=True)

    used_y_loss = []
    used_y_acc  = []
    for i, name in enumerate(model_names):
        color = colors[i]
        ls = line_styles[i % len(line_styles)]

        # valid loss
        vl = valid_losses_list[i]
        ax_loss_v.plot(epochs, vl, color=color, linestyle=ls, linewidth=2, label=name)
        min_v = min(vl); min_ep = vl.index(min_v) + 1
        ax_loss_v.scatter(min_ep, min_v, color=color, s=60, marker='*')
        _place_ann(ax_loss_v, min_ep, min_v, f"{name} {min_v:.4f}", color, i, used_y_loss)

        # valid acc
        va = valid_acc_list[i]
        ax_acc_v.plot(epochs, va, color=color, linestyle=ls, linewidth=2, label=name)
        max_v = max(va); max_ep = va.index(max_v) + 1
        ax_acc_v.scatter(max_ep, max_v, color=color, s=60, marker='o')
        _place_ann(ax_acc_v, max_ep, max_v, f"{name} {max_v:.3f}", color, i, used_y_acc)

    ax_loss_v.set_title("Valid Loss", fontsize=14)
    ax_loss_v.set_xlabel("Epoch")
    ax_loss_v.set_ylabel("Loss")
    ax_loss_v.grid(alpha=0.25)
    ax_loss_v.legend(fontsize=9, ncol=1)

    ax_acc_v.set_title("Valid Accuracy", fontsize=14)
    ax_acc_v.set_xlabel("Epoch")
    ax_acc_v.set_ylabel("Accuracy")
    ax_acc_v.grid(alpha=0.25)
    ax_acc_v.legend(fontsize=9, ncol=1)

    # plt.suptitle("Validation Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"./result/{save_prefix}_valid_{n_epochs}.png", dpi=300, bbox_inches='tight')
    plt.show()
def plot_confusion_matrix(
    all_targets,
    all_preds,
    save_path="confusion_matrix.png"
):
    """
    绘制完整混淆矩阵。

    Args:
        all_targets: List[int] or ndarray, 所有真实标签
        all_preds:   List[int] or ndarray, 所有模型预测标签
        save_path: 文件保存路径
    """
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(60, 50))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues"
    )
    plt.xlabel("Pred")
    plt.ylabel("Target")
    # plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return cm
def plot_confusion_matrix_subset(
    all_targets,
    all_preds,
    num_classes_to_show=20,
    selected_classes=None,
    save_path="cm_20classes.png"
):
    """
    绘制从全量混淆矩阵中抽取 20 个类别（或指定类别）的子矩阵。

    Args:
        all_targets: List[int] or ndarray, 所有真实标签
        all_preds:   List[int] or ndarray, 所有模型预测标签
        num_classes_to_show: int, 默认随机抽取 20 类
        selected_classes: List[int], 若不为 None，则使用指定的类
        save_path: 文件保存路径
    """
    
    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)

    # 计算完整 CM
    cm = confusion_matrix(all_targets, all_preds)
    total_classes = cm.shape[0]

    # 如果没指定类，则随机选
    if selected_classes is None:
        selected_classes = np.random.choice(total_classes, num_classes_to_show, replace=False)
        selected_classes = np.sort(selected_classes)
        print("随机选取的类:", selected_classes)
    else:
        print("使用指定的类:", selected_classes)

    # 截取子矩阵
    cm_sub = cm[np.ix_(selected_classes, selected_classes)]

    # 绘制图
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm_sub,
        annot=True,
        cmap="Blues",
        xticklabels=selected_classes,
        yticklabels=selected_classes
    )
    plt.xlabel("Pred")
    plt.ylabel("Target")
    # plt.title(f"Confusion Matrix ({len(selected_classes)} Classes)")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return cm_sub, selected_classes
