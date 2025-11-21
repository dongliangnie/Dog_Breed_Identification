import os 
import torch
import torch.nn as nn
import numpy as np

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda):
    """returns trained model"""
    model_name = model.__class__.__name__.lower()
    
    # 创建必要的目录
    os.makedirs("./model_weight", exist_ok=True)
    os.makedirs("./log/train_loss", exist_ok=True)
    os.makedirs("./log/valid_loss", exist_ok=True)
    
    save_models_path = "./model_weight/" + str(model_name) + '_epoch' + str(n_epochs) + '.pt'
    save_train_path = "./log/train_loss/" + str(model_name) + '_epoch' + str(n_epochs) + '.log'
    save_valid_path = "./log/valid_loss/" + str(model_name) + '_epoch' + str(n_epochs) + '.log'
    
    # 检查模型和损失文件是否存在
    if os.path.exists(save_models_path) and os.path.exists(save_train_path) and os.path.exists(save_valid_path):
        print('Model already exists, loading from', save_models_path)
        model.load_state_dict(torch.load(save_models_path))
        with open(save_train_path, 'r') as f:
            train_losses = [float(line.strip()) for line in f]
        with open(save_valid_path, 'r') as f:
            valid_losses = [float(line.strip()) for line in f]
        return model, train_losses, valid_losses
    
    # 初始化跟踪变量
    valid_loss_min = np.inf 
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, n_epochs + 1):
        # 初始化监控变量
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # 训练模型 #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # 移动到GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            # 计算损失并更新模型参数
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 移动平均
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx % 100 == 0:
                print('Epoch: %d \tBatch: %d \tTraining Loss: %.6f' % (epoch, batch_idx + 1, train_loss))
        
        train_loss_value = train_loss.cpu().item() if use_cuda else train_loss.item()
        train_losses.append(train_loss_value)
        
        ######################    
        # 验证模型 #
        ######################
        model.eval()
        with torch.no_grad():  # 添加no_grad以提高效率
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # 移动到GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                
                # 更新平均验证损失
                output = model(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        valid_loss_value = valid_loss.cpu().item() if use_cuda else valid_loss.item()
        valid_losses.append(valid_loss_value)
        
        # 打印训练/验证统计信息
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch, train_loss_value, valid_loss_value))
        
        # 如果验证损失减少，保存模型
        if valid_loss_value < valid_loss_min:
            torch.save(model.state_dict(), save_models_path)
            print('BOOM! Validation loss decreased ({:.4f} --> {:.4f}).  Saving model...'.format(
                valid_loss_min, valid_loss_value))
            valid_loss_min = valid_loss_value
    
    # 修正：正确保存损失数据
    with open(save_train_path, 'w') as f:
        for loss_value in train_losses:
            f.write(f"{loss_value}\n")
    
    with open(save_valid_path, 'w') as f:   
        for loss_value in valid_losses:
            f.write(f"{loss_value}\n")
    
    print(f"训练完成！模型保存至: {save_models_path}")
    print(f"训练损失保存至: {save_train_path}")
    print(f"验证损失保存至: {save_valid_path}")
    
    # 返回训练好的模型
    return model, train_losses, valid_losses
def transfer_train(model_transfer, dataloaders, lr=0.01, n_epochs=15,use_cuda=1):
    if use_cuda:
        model_transfer = model_transfer.cuda()
    # 冻结所有参数
    for param in model_transfer.parameters():
        param.requires_grad = False
    # 动态找到并替换最后的分类层
    new_layer = None
    if hasattr(model_transfer, 'fc'):  # ResNet, Inception等
        in_features = model_transfer.fc.in_features
        new_layer = nn.Linear(in_features, 120)
        model_transfer.fc = new_layer
    elif hasattr(model_transfer, 'classifier'):  # VGG, DenseNet等
        if isinstance(model_transfer.classifier, nn.Linear):
            in_features = model_transfer.classifier.in_features
            new_layer = nn.Linear(in_features, 120)
            model_transfer.classifier = new_layer
        else:
            # 如果classifier是Sequential，找到最后一个Linear层
            for name, module in model_transfer.classifier.named_children():
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    new_layer = nn.Linear(in_features, 120)
                    model_transfer.classifier = new_layer
                    break
    elif hasattr(model_transfer, 'last_linear'):  # SENet, EfficientNet等
        in_features = model_transfer.last_linear.in_features
        new_layer = nn.Linear(in_features, 120)
        model_transfer.last_linear = new_layer
    else:
        # 如果以上都不行，手动搜索最后一个Linear层
        for name, module in model_transfer.named_modules():
            if isinstance(module, nn.Linear):
                last_linear_name = name
                in_features = module.in_features
        
        # 通过字符串路径替换层
        if 'last_linear_name' in locals():
            parts = last_linear_name.split('.')
            current_module = model_transfer
            for part in parts[:-1]:
                current_module = getattr(current_module, part)
            new_layer = nn.Linear(in_features, 120)
            setattr(current_module, parts[-1], new_layer)
    
    if use_cuda and new_layer is not None:
        new_layer = new_layer.cuda()
        print("New classification layer moved to GPU")
    
    criterion_transfer = nn.CrossEntropyLoss()
    model_transfer_grad_parameters = filter(lambda p: p.requires_grad, model_transfer.parameters())
    optimizer_transfer = torch.optim.SGD(model_transfer_grad_parameters, lr=lr)
    model_name=model_transfer.__class__.__name__.lower()

    model_transfer,train_losses,valid_losses = train(n_epochs, dataloaders, model_transfer, optimizer_transfer, criterion_transfer, use_cuda)
    return model_transfer,train_losses,valid_losses