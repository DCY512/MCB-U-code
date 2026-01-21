# models/modules/distillation/utils.py
import torch
import torch.nn.functional as F

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    DKD (Decoupled Knowledge Distillation) 的核心实现。
    """
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # TCKD
    log_pred_student_tck = F.log_softmax(logits_student / temperature - 1000 * other_mask, dim=1)
    pred_teacher_tck = F.softmax(logits_teacher / temperature - 1000 * other_mask, dim=1)
    tckd_loss = F.kl_div(log_pred_student_tck, pred_teacher_tck, reduction='batchmean') * (temperature ** 2)

    # NCKD
    log_pred_student_nck = F.log_softmax(logits_student / temperature - 1000 * gt_mask, dim=1)
    pred_teacher_nck = F.softmax(logits_teacher / temperature - 1000 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_nck, pred_teacher_nck, reduction='batchmean') * (temperature ** 2)
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    """
    为多标签场景生成 ground-truth 掩码。
    target 本身就是一个 multi-hot 编码，我们只需要确保它是布尔型即可。
    """
    return target.bool()

def _get_other_mask(logits, target):
    """
    为多标签场景生成 "other" 掩码。
    """
    return ~target.bool() # ~ 是布尔类型的 "NOT" 操作