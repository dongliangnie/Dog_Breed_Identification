import torch
import torch.nn as nn
import torch.nn.functional as F
class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss (Hinton)
    teacher_logits: 老师的 logits
    student_logits: 学生 logits
    target: ground truth
    T: 温度系数
    alpha: teacher 软标签所占权重
    """
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, target):
        # Hard Loss（学生对真实标签的 CE）
        hard_loss = self.ce(student_logits, target)

        # Soft Loss（学生 vs 教师 softmax 后的分布）
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)

        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T * self.T)

        # 总损失
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss