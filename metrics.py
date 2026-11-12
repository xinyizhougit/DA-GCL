import numpy as np


def metrics(y, p, mask):
    valid = mask & np.isfinite(y)
    if np.any(valid & ~np.isfinite(p)):
        raise ValueError('Non-finite predictions on the shared evaluation mask')
    if not valid.any():
        return dict(MAE=None, RMSE=None, count=0)
    error = (y-p)[valid]
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.square(error).mean())), count=int(valid.sum()))


def all_horizons(y, p, mask):
    return dict(overall=metrics(y, p, mask), horizons={str(15*(h+1)): metrics(y[..., h], p[..., h], mask[..., h]) for h in range(y.shape[-1])})
