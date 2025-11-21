import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
# 在文件开头添加辅助函数
def safe_class_names(class_names):
    """安全处理 class_names，处理 pandas Index 和其他类型"""
    if class_names is None:
        return None
    if hasattr(class_names, 'tolist'):
        return class_names.tolist()
    return class_names

def safe_class_name_access(class_names, index, default_prefix="Class"):
    """安全访问类别名称"""
    if class_names is None:
        return f"{default_prefix} {index}"
    
    safe_names = safe_class_names(class_names)
    if safe_names is not None and index < len(safe_names):
        return safe_names[index]
    else:
        return f"{default_prefix} {index}"
def evaluate_multiclass_model(model, dataloader, use_cuda=True, class_names=None):
    """
    全面评估多分类模型
    
    Args:
        model: 训练好的模型
        dataloader: 测试数据加载器
        use_cuda: 是否使用GPU
        class_names: 类别名称列表
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in dataloader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    print("=== 多分类模型评价 ===")
    print(f"总样本数: {len(all_targets)}")
    print(f"类别数: {len(np.unique(all_targets))}")
    
    # 计算各项指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    print(f"\n=== 总体指标 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"加权精确率 (Precision): {precision:.4f}")
    print(f"加权召回率 (Recall): {recall:.4f}")
    print(f"加权F1分数: {f1:.4f}")
    
    # 详细分类报告
    print(f"\n=== 详细分类报告 ===")
    if class_names is not None and len(class_names) > 0:
        print(classification_report(all_targets, all_predictions, target_names=class_names))
    else:
        print(classification_report(all_targets, all_predictions))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_predictions, class_names)
    
    # 绘制ROC曲线（对于多分类）
    if len(np.unique(all_targets)) <= 10:  # 类别太多时ROC图会太复杂
        plot_multiclass_roc(all_targets, all_probabilities, class_names)
    
    # 类别级别分析
    analyze_per_class_performance(all_targets, all_predictions, all_probabilities, class_names)
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(60, 50)):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 打印混淆矩阵分析
    print("\n=== 混淆矩阵分析 ===")
    print(f"总体准确率: {accuracy:.4f}")
    
    # 计算每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\n各类别准确率:")
    for i, acc in enumerate(class_accuracies):
        class_name = class_names[i] if class_names else f'Class {i}'
        print(f"  {class_name}: {acc:.4f}")
def plot_multiclass_roc(y_true, y_prob, class_names=None, figsize=(10, 8)):
    """绘制多分类ROC曲线"""
    n_classes = y_prob.shape[1]
    
    # 安全处理 class_names
    safe_names = safe_class_names(class_names)
    if safe_names is None:
        safe_names = [f'Class {i}' for i in range(n_classes)]
    
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制ROC曲线
    plt.figure(figsize=figsize)
    
    # 绘制每个类别的ROC曲线
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        class_name = safe_names[i] if i < len(safe_names) else f'Class {i}'
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.4f})')
    
    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 打印AUC统计
    print("\n=== AUC统计 ===")
    for i in range(n_classes):
        class_name = safe_names[i] if i < len(safe_names) else f'Class {i}'
        print(f"  {class_name}: {roc_auc[i]:.4f}")
    print(f"  微平均AUC: {roc_auc['micro']:.4f}")
def analyze_per_class_performance(y_true, y_pred, y_prob, class_names=None):
    """分析每个类别的性能"""
    from sklearn.metrics import precision_recall_fscore_support
    
    n_classes = len(np.unique(y_true))
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 计算每个类别的准确率
    cm = confusion_matrix(y_true, y_pred)
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    
    print("\n=== 各类别详细性能 ===")
    print(f"{'类别':<20} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<8}")
    print("-" * 70)
    
    for i in range(n_classes):
        class_name = safe_class_name_access(class_names, i)
        print(f"{class_name:<20} {accuracy_per_class[i]:<8.4f} {precision_per_class[i]:<8.4f} "
              f"{recall_per_class[i]:<8.4f} {f1_per_class[i]:<8.4f} {support_per_class[i]:<8}")
    
    # 识别性能最好和最差的类别
    worst_f1_idx = np.argmin(f1_per_class)
    best_f1_idx = np.argmax(f1_per_class)
    
    print(f"\n=== 性能分析 ===")
    best_class_name = safe_class_name_access(class_names, best_f1_idx)
    worst_class_name = safe_class_name_access(class_names, worst_f1_idx)
    print(f"最佳性能类别: {best_class_name} (F1: {f1_per_class[best_f1_idx]:.4f})")
    print(f"最差性能类别: {worst_class_name} (F1: {f1_per_class[worst_f1_idx]:.4f})")
    
    # 识别样本数量不平衡问题
    max_samples = np.max(support_per_class)
    min_samples = np.min(support_per_class)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    print(f"样本不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  警告: 存在严重的类别不平衡问题")
def analyze_prediction_confidence(y_true, y_pred, y_prob, class_names=None):
    """分析预测置信度"""
    n_classes = y_prob.shape[1]
    
    # 计算正确和错误预测的置信度
    correct_mask = (y_true == y_pred)
    correct_confidences = y_prob[correct_mask].max(axis=1)
    wrong_confidences = y_prob[~correct_mask].max(axis=1)
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 置信度分布
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(wrong_confidences, bins=50, alpha=0.7, label='Wrong Predictions', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 准确率 vs 置信度
    plt.subplot(1, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    accuracy_per_bin = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (y_prob.max(axis=1) >= confidence_bins[i]) & (y_prob.max(axis=1) < confidence_bins[i+1])
        if np.sum(mask) > 0:
            accuracy = np.mean(y_true[mask] == y_pred[mask])
            accuracy_per_bin.append(accuracy)
        else:
            accuracy_per_bin.append(0)
    
    plt.plot(confidence_bins[:-1] + 0.05, accuracy_per_bin, 'o-', linewidth=2)
    plt.xlabel('Confidence Bin')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 置信度分析 ===")
    print(f"正确预测的平均置信度: {np.mean(correct_confidences):.4f}")
    print(f"错误预测的平均置信度: {np.mean(wrong_confidences):.4f}")
    print(f"校准误差: {np.abs(np.mean(correct_confidences) - np.mean(correct_confidences)):.4f}")
# 使用示例
def comprehensive_model_evaluation(model, test_loader, class_names, use_cuda=True):
    """完整的模型评估流程"""
    
    # 基础评估
    results = evaluate_multiclass_model(model, test_loader, use_cuda, class_names)
    
    # 置信度分析
    analyze_prediction_confidence(
        results['targets'], 
        results['predictions'], 
        results['probabilities'], 
        class_names
    )
    
    # 生成评估报告
    generate_evaluation_report(results, class_names)
    
    return results

def generate_evaluation_report(results, class_names):
    """生成评估报告"""
    print("\n" + "="*50)
    print("          模型评估总结报告")
    print("="*50)
    
    print(f"总体准确率: {results['accuracy']:.4f}")
    print(f"加权F1分数: {results['f1']:.4f}")
    
    # 性能等级判断
    if results['accuracy'] >= 0.9:
        performance_level = "优秀"
    elif results['accuracy'] >= 0.8:
        performance_level = "良好" 
    elif results['accuracy'] >= 0.7:
        performance_level = "一般"
    else:
        performance_level = "需要改进"
    
    print(f"性能等级: {performance_level}")
    
    # 建议
    print(f"\n建议:")
    if results['accuracy'] < 0.7:
        print("  • 考虑增加训练数据")
        print("  • 尝试数据增强")
        print("  • 调整模型架构")
    elif results['f1'] < results['accuracy'] - 0.1:
        print("  • 存在类别不平衡问题")
        print("  • 考虑使用加权损失函数")
    else:
        print("  • 模型性能良好，可以考虑部署")

