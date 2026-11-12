from pathlib import Path
from dataclasses import asdict
import json
import random
import time
import numpy as np
import torch
from model import DAGCL
from losses import intra_region, inter_region
from sampling import origins, batch, training_similarity
from graph_construction import structural_neighbors
from metrics import all_horizons


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_inputs(data, graph, c, origin, failure=None):
    x, y, mask = batch(data, c, int(origin))
    x = x.copy()
    ids = range(int(origin)-c.history+1, int(origin)+1)
    gs = [graph.at(t) for t in ids]
    active = np.stack([graph.active(t) for t in ids])
    if failure is not None:
        # 仅破坏历史输入/图，绝不删除失败站点的真实评价目标。
        x[:, failure, :] = 0
        active[:, failure] = False
    return torch.tensor(x, device=c.device), gs, torch.tensor(active, device=c.device), y, mask


def external_inputs(data, graph, c, origin_times):
    slots = np.asarray(origin_times.hour*4+origin_times.minute//15-c.start_hour*4)
    x = data['profile'][slots]
    activity = np.broadcast_to(data['service'][:data['split'][0]].any(0), x.shape[:2]).copy()
    return torch.tensor(x, device=c.device), [graph.static]*len(x), torch.tensor(activity, device=c.device)


def evaluate(model, data, graph, c, split='val', save=False, failure=None):
    model.eval()
    ys, ps, masks, used = [], [], [], []
    with torch.no_grad():
        for t in origins(data, c, split):
            x, gs, active, y, mask = sample_inputs(data, graph, c, t, failure)
            p, _, _ = model(x, gs, active)
            ys.append(y); ps.append(p.cpu().numpy()*data['scale']); masks.append(mask); used.append(int(t))
    y, p, mask = np.stack(ys), np.stack(ps), np.stack(masks)
    result = all_horizons(y, p, mask)
    if save:
        target_times = np.array([[str(data['times'][t+h]) for h in range(1, c.horizon+1)] for t in used])
        np.savez_compressed(Path(c.output, f'{split}_predictions.npz'), y_true=y, y_pred=p, mask=mask,
                            origins=used, nodes=np.array(data['nodes']), target_times=target_times)
        Path(c.output, f'{split}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_checkpoint(path, c, data):
    checkpoint = torch.load(path, map_location=c.device, weights_only=True)
    if checkpoint['nodes'] != data['nodes']:
        raise ValueError('Checkpoint node order mismatch')
    if not np.isclose(checkpoint['scale'], data['scale']):
        raise ValueError('Checkpoint training-only normalization mismatch')
    from config import Config
    saved = Config(**checkpoint['config']); saved.device = c.device
    for key in ('history', 'horizon', 'ablation', 'dtw_reference', 'dtw_band', 'dtw_sigma',
                'gmm_components', 'gmm_shrinkage', 'start_hour', 'zero_missing', 'dbc_minutes', 'topology_filter'):
        if getattr(saved, key) != getattr(c, key):
            raise ValueError(f'Checkpoint/input configuration mismatch: {key}')
    model = DAGCL(data['x'].shape[-1], saved).to(c.device)
    model.load_state_dict(checkpoint['model'])
    return model


def train(datasets, graphs, c, test=True):
    seed_everything(c.seed)
    root = Path(c.output); root.mkdir(parents=True, exist_ok=True)
    (root/'config.json').write_text(json.dumps(asdict(c), indent=2), encoding='utf-8')
    d, g = datasets[c.target], graphs[c.target]
    positive, negative = structural_neighbors(c.target, datasets)
    negative = negative[:c.negative_regions]
    if len(negative) == 0:
        raise ValueError('Contrastive loss needs at least one dissimilar external region')
    (root/'external_regions.json').write_text(json.dumps(dict(positive=positive, negative=negative), indent=2), encoding='utf-8')
    model = DAGCL(d['x'].shape[-1], c).to(c.device)
    if c.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(model.parameters(), lr=c.lr) if c.optimizer == 'adam' else torch.optim.SGD(model.parameters(), lr=c.lr)
    parameters = list(model.parameters())
    shared_ids = {id(p) for p in model.backbone.parameters()}
    shared = [id(p) in shared_ids for p in parameters]
    similarity = torch.tensor(training_similarity(d), device=c.device)
    train_origins = origins(d, c, 'train')
    rng = np.random.default_rng(c.seed)
    history, best, stale = [], float('inf'), 0
    for epoch in range(c.epochs):
        model.train(); rng.shuffle(train_origins)
        epoch_start = time.perf_counter(); grad_log = []; train_losses = []
        for begin in range(0, len(train_origins), c.batch_size):
            ids = train_origins[begin:begin+c.batch_size]
            denominator = sum(int(batch(d, c, t)[2].sum()) for t in ids)
            if denominator == 0:
                continue
            # 分别累计两任务的完整逻辑批次梯度，再按共享梯度范数计算统一lambda。
            # 这与逐微批独立计算lambda不同；无二阶梯度、无近似梯度范数。
            gp = [torch.zeros_like(p) for p in parameters]
            gc = [torch.zeros_like(p) for p in parameters]
            losses = 0.
            for t in ids:
                x, gs, active, y, mask = sample_inputs(d, g, c, t)
                enabled = c.ablation != 'no_cl'
                pred, za, zp = model(x, gs, active, contrastive=enabled)
                y = torch.tensor(y/d['scale'], device=c.device)
                mask = torch.tensor(mask, device=c.device)
                pred_loss = ((pred-y).square()*mask).sum()/denominator
                cl = pred.sum()*0
                if enabled and active[-1].any():
                    cl = intra_region(za[None], zp[None], similarity, active[-1][None], c.temperature, c.false_negative_phi)
                    pool = (za*active[-1, :, None]).sum(0)/active[-1].sum().clamp_min(1)
                    # 冻结的是外区训练资料，不冻结共享编码器的可学习权重。
                    proto = {}
                    times = d['times'][int(t)-c.history+1:int(t)+1]
                    for r in positive+negative:
                        xx, gg, aa = external_inputs(datasets[r], graphs[r], c, times)
                        proto[r] = model.prototype(xx, gg, aa)[None]
                    cl = cl+c.gamma*inter_region(pool[None], [proto[r] for r in positive], [proto[r] for r in negative], c.temperature)
                cl = cl/len(ids)
                pgrads = torch.autograd.grad(pred_loss, parameters, retain_graph=enabled, allow_unused=True)
                cgrads = torch.autograd.grad(cl, parameters, allow_unused=True) if enabled else [None]*len(parameters)
                for j, (pgrad, cgrad) in enumerate(zip(pgrads, cgrads)):
                    if pgrad is not None:
                        gp[j].add_(pgrad.detach())
                    if cgrad is not None:
                        gc[j].add_(cgrad.detach())
                losses += float(pred_loss.detach())
            p2 = sum(v.square().sum() for v, flag in zip(gp, shared) if flag)
            q2 = sum(v.square().sum() for v, flag in zip(gc, shared) if flag)
            dot = sum((p*q).sum() for p, q, flag in zip(gp, gc, shared) if flag)
            pn, qn = p2.sqrt(), q2.sqrt()
            weight = (c.balance_prior*pn/(qn+c.eps)).clamp(max=c.lambda_max).detach()
            if c.ablation == 'static_weight':
                weight = weight.new_tensor(c.static_cl_weight)
            if c.ablation == 'no_cl':
                weight = weight*0
            opt.zero_grad(set_to_none=True)
            for p, a, b in zip(parameters, gp, gc):
                p.grad = a+weight*b+2*c.eta*p.detach()
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            opt.step()
            grad_log.append(dict(pred_norm=float(pn), cl_norm=float(qn), lambda_cl=float(weight),
                                 cosine=float(dot/(pn*qn+c.eps))))
            train_losses.append(losses)
        if c.device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = time.perf_counter()-epoch_start
        result = evaluate(model, d, g, c)
        val = result['overall']['MAE']
        if val is None or not np.isfinite(val):
            raise ValueError('No finite validation MAE')
        row = dict(epoch=epoch+1, training_normalized_mse=float(np.mean(train_losses)),
                   validation_raw_mae=val, training_seconds=seconds, gradient_log=grad_log)
        history.append(row); print({k: v for k, v in row.items() if k != 'gradient_log'}, flush=True)
        if val < best:
            best, stale = val, 0
            torch.save(dict(model=model.state_dict(), config=asdict(c), nodes=d['nodes'], scale=d['scale']), root/'best.pt')
        else:
            stale += 1
        if stale >= c.patience:
            break
    (root/'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    model = load_checkpoint(root/'best.pt', c, d)
    evaluate(model, d, g, c, 'val', save=True)
    if test:
        evaluate(model, d, g, c, 'test', save=True)
    (root/'resources.json').write_text(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
         cuda_peak_bytes=torch.cuda.max_memory_allocated() if c.device.startswith('cuda') else None,
         logical_batch=c.batch_size, physical_execution='one sample at a time; exact logical-batch task gradients'), indent=2), encoding='utf-8')
    return best
