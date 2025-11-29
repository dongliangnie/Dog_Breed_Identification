import torch 
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from visualize.plot_loss import plot_confusion_matrix,plot_confusion_matrix_subset
@torch.no_grad()
def evaluate_feature_fusion(
    model, dataloader, use_cuda=False,save_path=('./result/fusion_cm.png','./result/fusion_cm_rnd20.png')
):
    if use_cuda:
        model = model.cuda()
    model.eval()
    all_targets = []
    all_preds = []

    for images, targets in tqdm(dataloader, desc='Fusion Ensemble Inference'):
        if use_cuda:
            images = images.cuda()
            targets = targets.cuda()
        output = model(images)
        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    # 计算 ACC
    acc = np.mean(np.array(all_targets) == np.array(all_preds)) 
    # 混淆矩阵
    plot_confusion_matrix(all_targets, all_preds,save_path=save_path[0])
    plot_confusion_matrix_subset(all_targets, all_preds,save_path=save_path[1])
    return acc, all_preds, all_targets