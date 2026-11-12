import math
import torch
from torch import nn
from torch.nn import functional as F


def sparse_normalized(graph, active, device):
    # 对称化与自环严格先于归一化；物理图保持稀疏矩阵-特征乘法。
    a = graph.maximum(graph.T).tocoo()
    ids = torch.tensor([a.row.tolist(), a.col.tolist()], dtype=torch.long, device=device)
    val = torch.tensor(a.data, dtype=torch.float32, device=device)
    keep = active[ids[0]] & active[ids[1]]
    ids, val = ids[:, keep], val[keep]
    nodes = torch.where(active)[0]
    ids = torch.cat([ids, torch.stack([nodes, nodes])], dim=1)
    val = torch.cat([val, torch.ones(len(nodes), device=device)])
    matrix = torch.sparse_coo_tensor(ids, val, (len(active), len(active))).coalesce()
    ids, val = matrix.indices(), matrix.values()
    degree = torch.zeros(len(active), device=device).scatter_add_(0, ids[0], val)
    inv = degree.clamp_min(1).rsqrt()
    return torch.sparse_coo_tensor(ids, val*inv[ids[0]]*inv[ids[1]], matrix.shape).coalesce()


def dense_scaled_laplacian(a, active):
    # lambda_max取归一化拉普拉斯的上界2，因此缩放后L~= -D^-1/2 A D^-1/2。
    mask = active[:, None] & active[None, :]
    a = torch.maximum(a, a.T)*mask
    a = a+torch.diag(active.to(a.dtype))
    degree = a.sum(-1).clamp_min(1e-8)
    inv = degree.rsqrt()
    return -(inv[:, None]*a*inv[None, :])


def chebyshev_apply(operator, x, order, sparse=False):
    # 只递推T_k(L)X，避免构造矩阵多项式T_k(L)。
    output = [x]
    multiply = (lambda z: torch.sparse.mm(operator, z)) if sparse else (lambda z: operator @ z)
    if order >= 1:
        output.append(multiply(x))
    for _ in range(2, order+1):
        output.append(2*multiply(output[-1])-output[-2])
    return output


class EvolvingFilter(nn.Module):
    def __init__(self, features, c):
        super().__init__(); self.c = c
        self.context = nn.Linear(features, c.cheb_order+1)
        self.gru = nn.GRUCell(c.cheb_order+1, c.cheb_order+1)
        self.fixed = nn.Parameter(torch.ones(c.cheb_order+1)/(c.cheb_order+1))

    def forward(self, x, graph, active, previous):
        norm = sparse_normalized(graph, active, x.device)
        operator = torch.sparse_coo_tensor(norm.indices(), -norm.values(), norm.shape).coalesce()
        pooled = (x*active[:, None]).sum(0)/active.sum().clamp_min(1)
        theta = self.gru(self.context(pooled), previous)
        terms = chebyshev_apply(operator, x, self.c.cheb_order, sparse=True)
        if self.c.ablation == 'no_filter':
            return x, theta
        if self.c.ablation == 'static_gft':
            # 静态热核GFT消融；明确采用h(lambda)=exp(-lambda)，论文未给其截止频率。
            laplacian = torch.eye(len(x), device=x.device)+operator.to_dense()
            vals, basis = torch.linalg.eigh(laplacian)
            return basis @ (torch.exp(-vals)[:, None]*(basis.T @ x)), theta
        coefficients = self.fixed if self.c.ablation == 'static_cheb' else theta
        return sum(w*z for w, z in zip(coefficients, terms)), theta


class UnrolledSER(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c

    def forward(self, history, active):
        n = len(history)
        z = history.new_zeros(n, n)
        mask = active[:, None] & active[None, :]
        mask.fill_diagonal_(False)
        # Frobenius上界保证ISTA步长不大于1/||XX^T||_2，避免显式特征分解。
        step = 1/(history.detach().square().sum()+1e-6)
        if self.c.ablation == 'no_ser':
            return history, z
        for _ in range(self.c.ista_steps):
            gradient = (z @ history-history) @ history.T
            proposal = z-step*gradient
            z = proposal.sign()*(proposal.abs()-step*self.c.lambda_ser).clamp_min(0)*mask
        return z @ history, z


class SpatialLayer(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        h = c.hidden
        self.weights = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(c.cheb_order+1)])
        self.hops = nn.Linear(1, c.cheb_order+1)
        self.fusion = nn.Linear(2*h, h)
        self.gate = nn.Linear(2*h, h)
        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, h, adjacency, active):
        laplacian = dense_scaled_laplacian(adjacency, active)
        density = (adjacency > 0).float().mean().reshape(1)
        omega = torch.sigmoid(self.hops(density))
        terms = chebyshev_apply(laplacian, h, self.c.cheb_order)
        aggregated = F.gelu(sum(w*layer(z) for w, layer, z in zip(omega, self.weights, terms)))
        joined = torch.cat([h, aggregated], -1)
        out = self.norm(h+torch.sigmoid(self.gate(joined))*self.dropout(self.fusion(joined)))
        return out*active[:, None]


def degree_augment(adjacency, active, c):
    a = torch.maximum(adjacency, adjacency.T)
    degree = a.sum(-1)
    mean = degree[active].mean() if active.any() else degree.sum()*0
    probability = (2*mean/(degree[:, None]+degree[None, :]+c.eps)*c.edge_drop).clamp(c.drop_min, c.drop_max)
    if c.ablation == 'random_mask':
        probability = torch.full_like(probability, c.edge_drop)
    # 对称边共享一个伯努利样本，防止再对称化后改变实际drop概率。
    dropped = torch.triu(torch.rand_like(a) < probability, diagonal=1)
    dropped = dropped | dropped.T
    return a*(~dropped)


class Backbone(nn.Module):
    def __init__(self, features, c):
        super().__init__(); self.c = c; self.features = features
        self.eft = EvolvingFilter(features, c)
        self.ser = UnrolledSER(c)
        self.input = nn.Linear(features, c.hidden)
        self.query = nn.Linear(features, c.hidden, bias=False)
        self.key = nn.Linear(features, c.hidden, bias=False)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(c.beta/(1-c.beta))))
        self.layers = nn.ModuleList([SpatialLayer(c) for _ in range(c.spatial_layers)])

    def forward(self, x, graphs, active, perturb=False):
        sequence, recon_history = [], []
        theta = x.new_zeros(self.c.cheb_order+1)
        for t in range(len(x)):
            mask = active[t]
            value = x[t]*mask[:, None]
            filtered, theta = self.eft(value, graphs[t], mask, theta)
            recon_history.append(filtered)
            prefix = recon_history[-self.c.ser_window:]
            # 有限历史特征向量构造X_t，窗口左侧补零，不使用未来。
            padded = [torch.zeros_like(value)]*(self.c.ser_window-len(prefix))+prefix
            temporal_features = torch.cat(padded, dim=-1)
            reconstructed, z = self.ser(temporal_features, mask)
            features = reconstructed[:, -self.features:]
            q, k = self.query(features), self.key(features)
            logits = q @ k.T/math.sqrt(self.c.hidden)
            support = mask[:, None] & mask[None, :]
            if self.c.ser_guidance == 'support' and self.c.ablation != 'no_ser':
                support &= z.abs() > 1e-8
            logits = logits.masked_fill(~support, -torch.inf)
            valid_row = support.any(-1, keepdim=True)
            logits = torch.where(valid_row, logits, torch.zeros_like(logits))
            semantic = torch.softmax(logits, dim=-1)*support
            phy = torch.tensor(graphs[t].toarray(), device=x.device)
            phy = phy/(phy.sum(-1, keepdim=True).clamp_min(1))
            beta = torch.sigmoid(self.beta_logit)
            adjacency = beta*phy+(1-beta)*semantic
            if self.c.ablation == 'physical_only':
                adjacency = phy
            if perturb:
                adjacency = degree_augment(adjacency, mask, self.c)
                features = features*(torch.rand_like(features) >= self.c.feature_drop)
            h = self.input(features)
            for layer in self.layers:
                h = layer(h, adjacency, mask)
            sequence.append(h)
        return torch.stack(sequence)


class CausalGLU(nn.Module):
    def __init__(self, h, kernel, dilation, dropout):
        super().__init__(); self.padding = (kernel-1)*dilation
        self.conv = nn.Conv1d(h, 2*h, kernel, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x+self.dropout(F.glu(self.conv(F.pad(x, (self.padding, 0))), dim=1))


class DAGCL(nn.Module):
    def __init__(self, features, c):
        super().__init__(); self.c = c
        self.backbone = Backbone(features, c)
        self.temporal = nn.Sequential(*[CausalGLU(c.hidden, c.tcn_kernel, 2**i, c.dropout) for i in range(c.tcn_layers)])
        self.predictor = nn.Sequential(nn.Linear(c.hidden, 2*c.hidden), nn.GELU(), nn.Linear(2*c.hidden, c.horizon))
        self.projector = nn.Sequential(nn.Linear(c.hidden, c.hidden), nn.GELU())

    def forward(self, x, graphs, active, contrastive=False):
        anchor = self.backbone(x, graphs, active)
        temporal = self.temporal(anchor.permute(1, 2, 0))[:, :, -1]
        prediction = self.predictor(temporal)
        za = self.projector(anchor[-1])*active[-1, :, None]
        zp = None
        if contrastive:
            perturbed = self.backbone(x, graphs, active, perturb=True)
            zp = self.projector(perturbed[-1])*active[-1, :, None]
        return prediction, za, zp

    def prototype(self, x, graphs, active):
        # 外部输入只来自训练期日剖面；共享参数随训练更新，不读取验证/测试外区客流。
        representation = self.backbone(x, graphs, active)
        z = self.projector(representation[-1])
        return (z*active[-1, :, None]).sum(0)/active[-1].sum().clamp_min(1)
