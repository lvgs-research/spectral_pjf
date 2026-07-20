"""PC1 (exposure count/DC axis) ablation helpers: project the top right-singular vector out of the
pre-decoder pair embeddings, re-apply the trained decoder, compare ranking baseline vs PC1-removed."""
import numpy as np, torch
torch.set_num_threads(4)
from sklearn.metrics import roc_auc_score


def joint_score(decoder, Zm, Zj, pf):
    with torch.no_grad():
        a, c = decoder.forward_heads(torch.as_tensor(Zm, dtype=torch.float32),
                                     torch.as_tensor(Zj, dtype=torch.float32), pf)
        return (torch.nn.functional.logsigmoid(a) + torch.nn.functional.logsigmoid(c)).numpy()


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def gauc(groups, y, s, mask=None):
    """mean per-group AUC over groups with >=1 pos & >=1 neg; optional group-level `mask`."""
    groups = np.asarray(groups); y = np.asarray(y); s = np.asarray(s)
    aucs = []
    for g in np.unique(groups):
        idx = groups == g
        if mask is not None and not mask[idx].any():
            continue
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        aucs.append(roc_auc_score(yy, s[idx]))
    return (np.mean(aucs) if aucs else np.nan), len(aucs)


def top_rvs(M, idxs):
    """unit top right-singular vectors (k, d) of centered M at the given PC indices."""
    Mc = np.asarray(M, np.float64) - np.asarray(M, np.float64).mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    V = Vt[list(idxs)]
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def proj_out_multi(Z, V):
    """Remove the orthonormal directions V (k, d) from every row of Z."""
    Z = np.asarray(Z, np.float64)
    return Z - (Z @ V.T) @ V
