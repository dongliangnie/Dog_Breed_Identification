# 可视化feature map
import numpy as np
import torch
def extract_features_map(model, dataloader, use_cuda=True, n_samples=1000, layer_name='fc'):
    """
    提取模型的最后一个线性层的输入特征
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        use_cuda: 是否使用GPU
        n_samples: 最大样本数
        layer_name: 要提取特征的层名称
    """
    # 将模型设置为评估模式
    model.eval()
    # 存储特征和标签
    features = []
    labels = []
    print("开始提取特征...")
    # 创建钩子来获取中间层输出
    feature_maps = {}
    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook
    # 注册钩子 - 针对不同模型结构可能需要调整
    model.avg_pool.register_forward_hook(get_features('avg_pool'))
    target_layer = 'avg_pool'
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if len(features) >= n_samples:
                break        
            if use_cuda:
                data = data.cuda() 
            # 前向传播
            _ = model(data)   
            # 获取特征
            if target_layer in feature_maps:
                batch_features = feature_maps[target_layer]   
                # 如果特征是多维的，展平
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.view(batch_features.size(0), -1)
                
                features.append(batch_features.cpu().numpy())
                labels.extend(target.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {len(features)} 个样本...")
    # 合并所有特征
    if not features:
        print("没有提取到特征，请检查层名称")
        return
    features_array = np.vstack(features) #一行一行拼接
    labels_array = np.array(labels) 
    
    print(f"特征形状: {features_array.shape}")
    print(f"标签形状: {labels_array.shape}")
    print(f"唯一标签: {np.unique(labels_array)}")
    return features_array,labels_array