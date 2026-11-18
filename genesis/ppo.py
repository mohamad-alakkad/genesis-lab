"""Minimal PPO in NumPy (float64): separate tanh MLPs for policy and value, manual backprop, Adam.

No pretrained weights: parameters come only from `init_params(seed)`.
"""
from __future__ import annotations

import numpy as np


def init_params(obs_dim, n_actions, hidden, seed):
    rng = np.random.default_rng(seed)

    def lin(i, o, gain):  # orthogonal init (standard for PPO)
        a = rng.standard_normal((max(i, o), min(i, o)))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))
        w = q if q.shape == (i, o) else q.T
        return gain * w[:i, :o], np.zeros(o)

    p = {}
    for net, out, g_out in (("pi", n_actions, 0.01), ("v", 1, 1.0)):
        dims = [obs_dim] + list(hidden)
        for k in range(len(hidden)):
            p[f"{net}.W{k}"], p[f"{net}.b{k}"] = lin(dims[k], dims[k + 1], np.sqrt(2))
        p[f"{net}.W{len(hidden)}"], p[f"{net}.b{len(hidden)}"] = lin(dims[-1], out, g_out)
    return p


def n_layers(p, net):
    return sum(1 for k in p if k.startswith(net + ".W"))


def mlp_forward(p, net, x):
    L = n_layers(p, net)
    hs = [x]
    h = x
    for k in range(L - 1):
        h = np.tanh(h @ p[f"{net}.W{k}"] + p[f"{net}.b{k}"])
        hs.append(h)
    out = h @ p[f"{net}.W{L - 1}"] + p[f"{net}.b{L - 1}"]
    return out, hs


def mlp_backward(p, net, hs, dout, grads):
    L = n_layers(p, net)
    g = dout
    for k in range(L - 1, -1, -1):
        grads[f"{net}.W{k}"] = hs[k].T @ g
        grads[f"{net}.b{k}"] = g.sum(0)
        if k > 0:
            g = (g @ p[f"{net}.W{k}"].T) * (1.0 - hs[k] ** 2)


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim); self.var = np.ones(dim); self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


class Adam:
    def __init__(self, params, lr):
        self.lr = lr; self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, b1=0.9, b2=0.999, eps=1e-5):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + eps)


def policy_logits(p, x):
    return mlp_forward(p, "pi", x)[0]


def ppo_loss_and_grads(p, obs, act, old_logp, adv, ret, clip, ent_coef, vf_coef):
    N = obs.shape[0]
    grads = {}
    logits, hs_pi = mlp_forward(p, "pi", obs)
    lp = log_softmax(logits); pr = np.exp(lp)
    logp = lp[np.arange(N), act]
    ratio = np.exp(logp - old_logp)
    s1 = ratio * adv; s2 = np.clip(ratio, 1 - clip, 1 + clip) * adv
    pg_loss = -np.mean(np.minimum(s1, s2))
    ent = -(pr * lp).sum(1)
    # gradient of -min(s1,s2) w.r.t. logp: active only where the unclipped term is the min
    active = s1 <= s2
    dlogp = np.where(active, -adv * ratio, 0.0) / N
    onehot = np.zeros_like(pr); onehot[np.arange(N), act] = 1.0
    dlogits = dlogp[:, None] * (onehot - pr)
    dH = -pr * (lp + ent[:, None])  # dH/dlogits
    dlogits += -ent_coef * dH / N
    mlp_backward(p, "pi", hs_pi, dlogits, grads)
    v, hs_v = mlp_forward(p, "v", obs)
    v = v[:, 0]
    v_loss = 0.5 * np.mean((v - ret) ** 2)
    dv = (vf_coef * (v - ret) / N)[:, None]
    mlp_backward(p, "v", hs_v, dv, grads)
    total = pg_loss - ent_coef * ent.mean() + vf_coef * v_loss
    stats = {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent.mean(),
             "clipfrac": float(np.mean(np.abs(ratio - 1) > clip)), "approx_kl": float(np.mean(old_logp - logp))}
    return total, grads, stats


def clip_grads(grads, max_norm):
    norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / (norm + 1e-12)
    return norm
