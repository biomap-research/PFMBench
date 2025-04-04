import torch

def spearman_correlation(y_true, y_pred):
    # 排序并获取秩
    rank_true = torch.argsort(torch.argsort(y_true))
    rank_pred = torch.argsort(torch.argsort(y_pred))
    
    # 计算秩差的平方
    d_squared = torch.pow(rank_true - rank_pred, 2).float()
    
    # 样本数量
    n = y_true.size(0)
    
    # 计算 Spearman 相关系数
    spearman_corr = 1 - (6 * torch.sum(d_squared)) / (n * (n**2 - 1))
    return spearman_corr.item()