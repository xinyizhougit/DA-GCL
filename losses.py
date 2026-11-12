import torch
from torch.nn import functional as F


def intra_region(anchor, perturbed, similarity, active, temperature, phi, chunk=256):
    # 正对为同站不同视图；高相关站点取消负样本惩罚；分块不改变InfoNCE。
    values = []
    for b in range(len(anchor)):
        ids = torch.where(active[b])[0]
        if len(ids) == 0:
            continue
        a, p = F.normalize(anchor[b, ids], dim=-1), F.normalize(perturbed[b, ids], dim=-1)
        allowed = similarity[ids[:, None], ids[None, :]] < phi
        allowed.fill_diagonal_(True)
        for begin in range(0, len(ids), chunk):
            end = min(begin+chunk, len(ids))
            logits = (a[begin:end] @ p.T)/temperature
            logits = logits.masked_fill(~allowed[begin:end], -torch.inf)
            labels = torch.arange(begin, end, device=a.device)
            values.append(F.cross_entropy(logits, labels, reduction='sum'))
    count = active.sum().clamp_min(1)
    return sum(values)/count if values else anchor.sum()*0


def inter_region(anchor, positives, negatives, temperature):
    a = F.normalize(anchor, dim=-1)
    neg = torch.stack([F.normalize(z, dim=-1) for z in negatives], dim=1)
    neg_logits = torch.einsum('bd,bmd->bm', a, neg)/temperature
    result = a.sum()*0
    for pos in positives:
        score = (a*F.normalize(pos, dim=-1)).sum(-1, keepdim=True)/temperature
        logits = torch.cat([score, neg_logits], dim=-1)
        result = result+F.cross_entropy(logits, torch.zeros(len(a), dtype=torch.long, device=a.device))
    return result


def balance_weight(pred, contrastive, parameters, c):
    # 按共享编码器参数的两项梯度范数计算lambda，裁剪并detach，避免二阶优化。
    parameters = list(parameters)
    gp = torch.autograd.grad(pred, parameters, retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(contrastive, parameters, retain_graph=True, allow_unused=True)
    p2, q2, dot = pred.new_zeros(()), pred.new_zeros(()), pred.new_zeros(())
    for p, q in zip(gp, gc):
        if p is not None:
            p2 += p.detach().square().sum()
        if q is not None:
            q2 += q.detach().square().sum()
        if p is not None and q is not None:
            dot += (p.detach()*q.detach()).sum()
    pn, qn = p2.sqrt(), q2.sqrt()
    weight = (c.balance_prior*pn/(qn+c.eps)).clamp(max=c.lambda_max).detach()
    return weight, dict(pred_grad_norm=float(pn), cl_grad_norm=float(qn),
                        cosine=float(dot/(pn*qn+c.eps)), cl_weight=float(weight))
