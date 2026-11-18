"""GENESIS world v1 — batch-vectorized NumPy reference implementation (float64).

All world randomness comes from a per-env mulberry32 stream (portable to JS). The JS port in
web/ must follow exactly the same draw order; see docs/WORLD_SPEC.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

C = json.loads((Path(__file__).with_name("world_constants.json")).read_text())
SPEC_VERSION = C["world_spec_version"]
N_ACTIONS = len(C["actions"])
OBS_DIM = C["n_rays"] * 3 + 3 + N_ACTIONS
U32 = np.uint32


class Mulberry32:
    """Vectorized mulberry32; next(mask) advances only masked streams."""

    def __init__(self, n: int):
        self.s = np.zeros(n, dtype=U32)

    def seed(self, mask: np.ndarray, seeds: np.ndarray):
        self.s[mask] = np.asarray(seeds, dtype=np.uint64).astype(U32)

    def next(self, mask: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            s = self.s + U32(0x6D2B79F5)
            self.s = np.where(mask, s, self.s)
            t = (s ^ (s >> U32(15))) * (s | U32(1))
            t = (t + ((t ^ (t >> U32(7))) * (t | U32(61)))) ^ t
            r = (t ^ (t >> U32(14))).astype(np.float64) / 4294967296.0
        return np.where(mask, r, 0.0)


class World:
    def __init__(self, n_envs: int, family: str = "train", consts: dict | None = None):
        self.c = c = dict(consts or C)
        self.f = c["families"][family]
        self.family = family
        E, MO, NF, NV, R = n_envs, c["max_obstacles"], c["max_food"], c["max_vents"], c["n_rays"]
        self.E = E
        self.rng = Mulberry32(E)
        self.pos = np.zeros((E, 2)); self.heading = np.zeros(E); self.energy = np.zeros(E)
        self.t = np.zeros(E, dtype=np.int64); self.last_action = np.full(E, 5, dtype=np.int64)
        self.contact = np.zeros(E, dtype=bool); self.in_hazard = np.zeros(E, dtype=bool)
        self.obs_pos = np.zeros((E, MO, 2)); self.obs_r = np.zeros((E, MO)); self.obs_on = np.zeros((E, MO), bool)
        self.food_pos = np.zeros((E, NF, 2))
        self.food_on = np.zeros((E, NF), bool)
        self.vent_pos = np.zeros((E, NV, 2)); self.vent_r = np.zeros((E, NV)); self.vent_on = np.zeros((E, NV), bool)
        self.vent_present = np.zeros((E, NV), bool); self.vent_timer = np.zeros((E, NV), dtype=np.int64)
        # episode statistics
        self.ep_food = np.zeros(E, np.int64); self.ep_hazard = np.zeros(E, np.int64)
        self.ep_dist = np.zeros(E); self.ep_contact = np.zeros(E, np.int64); self.ep_return = np.zeros(E)
        self.events = []  # (env, tick, kind, index) — food eaten etc., for demo/parity
        rel = np.linspace(-c["fov"] / 2, c["fov"] / 2, R)
        self.ray_rel = rel

    # ----------------------------------------------------------------- sampling helpers
    def _u(self, mask):
        return self.rng.next(mask)

    def _sample_point(self, mask, margin, clear_fn):
        """Candidate loop: per env draw (x, y) until clear_fn accepts, max attempts. Returns pts, ok."""
        W, H = self.c["width"], self.c["height"]
        need = mask.copy()
        out = np.zeros((self.E, 2)); ok = np.zeros(self.E, bool)
        margin = np.broadcast_to(np.asarray(margin, dtype=np.float64), (self.E,))
        for _ in range(self.c["sample_attempts"]):
            if not need.any():
                break
            x = margin + self._u(need) * (W - 2 * margin)
            y = margin + self._u(need) * (H - 2 * margin)
            p = np.stack([x, y], 1)
            out[need] = p[need]
            good = need & clear_fn(p)
            ok |= good
            need &= ~good
        return out, ok

    def _dist_obstacles_ok(self, p, extra):
        d = np.linalg.norm(self.obs_pos - p[:, None, :], axis=2)
        ex = np.asarray(extra, np.float64)
        ex = ex[:, None] if ex.ndim == 1 else ex
        return np.all(~self.obs_on | (d > self.obs_r + ex), 1)

    def _not_in_vents(self, p, extra=0.0):
        d = np.linalg.norm(self.vent_pos - p[:, None, :], axis=2)
        return np.all(~self.vent_on | (d > self.vent_r + extra), 1)

    def _vent_duration(self, mask, active):
        u = self._u(mask)
        return np.where(active, 150 + np.floor(u * 250), 100 + np.floor(u * 200)).astype(np.int64)

    def _place_food(self, mask, i):
        rc, rf = self.c["creature_radius"], self.c["food_radius"]
        def clear(p):
            return (self._dist_obstacles_ok(p, rf + 0.1) & self._not_in_vents(p)
                    & (np.linalg.norm(p - self.pos, axis=1) > 1.0))
        pts, _ = self._sample_point(mask, 0.4, clear)
        self.food_pos[mask, i] = pts[mask]

    # ----------------------------------------------------------------- reset
    def reset(self, mask: np.ndarray, seeds: np.ndarray):
        c, f = self.c, self.f
        mask = np.asarray(mask, bool)
        if not mask.any():
            return
        self.rng.seed(mask, np.asarray(seeds)[mask] if np.size(seeds) == self.E else seeds)
        m = mask
        self.obs_on[m] = False; self.obs_r[m] = 0.0; self.obs_pos[m] = 0.0
        self.vent_on[m] = False; self.vent_present[m] = False; self.vent_r[m] = 0.0; self.vent_pos[m] = -100.0
        self.food_on[m] = False; self.food_pos[m] = -100.0
        self.pos[m] = -100.0
        n_obs = f["obs_min"] + np.floor(self._u(m) * (f["obs_max"] - f["obs_min"] + 1)).astype(np.int64)
        for j in range(c["max_obstacles"]):
            mj = m & (j < n_obs)
            if not mj.any():
                continue
            r = f["obs_rmin"] + self._u(mj) * (f["obs_rmax"] - f["obs_rmin"])
            gap = 2 * c["creature_radius"] + 0.05  # creature can always pass between slates and walls
            pts, _ = self._sample_point(mj, r + gap, lambda p: self._dist_obstacles_ok(p, r + gap))
            self.obs_pos[mj, j] = pts[mj]; self.obs_r[mj, j] = r[mj]; self.obs_on[mj, j] = True
        for v in range(c["max_vents"]):
            mv = m & (v < f["n_vents"])
            if not mv.any():
                continue
            r = 0.6 + self._u(mv) * 0.4
            pts, _ = self._sample_point(mv, r, lambda p: self._dist_obstacles_ok(p, 0.0))
            active = self._u(mv) < 0.7
            self.vent_pos[mv, v] = pts[mv]; self.vent_r[mv, v] = r[mv]; self.vent_present[mv, v] = True
            self.vent_on[mv, v] = active[mv]
            self.vent_timer[mv, v] = self._vent_duration(mv, active)[mv]
        rc = c["creature_radius"]
        def clear_c(p):
            dv = np.linalg.norm(self.vent_pos - p[:, None, :], axis=2)
            return self._dist_obstacles_ok(p, rc + 0.1) & np.all(~self.vent_present | (dv > self.vent_r + rc), 1)
        pts, _ = self._sample_point(m, rc + 0.1, clear_c)
        self.pos[m] = pts[m]
        self.heading[m] = self._u(m)[m] * 2 * math.pi
        for i in range(f["food_n"]):
            self._place_food(m, i)
            self.food_on[m, i] = True
        self.energy[m] = c["energy_start"]; self.t[m] = 0; self.last_action[m] = 5
        self.contact[m] = False; self.in_hazard[m] = False
        self.ep_food[m] = 0; self.ep_hazard[m] = 0; self.ep_dist[m] = 0; self.ep_contact[m] = 0; self.ep_return[m] = 0

    # ----------------------------------------------------------------- visitor / perturbation mutations
    def insert_obstacle(self, mask, r=None, at=None):
        """Insert one slate into the first free slot (max_obstacles). Never on the creature or food."""
        c, f = self.c, self.f
        mask = mask & ~np.all(self.obs_on, 1)
        if not mask.any():
            return np.zeros(self.E, bool)
        if r is None:
            r = f["obs_rmin"] + self._u(mask) * (f["obs_rmax"] - f["obs_rmin"])
        r = np.broadcast_to(np.asarray(r, np.float64), (self.E,))
        def clear(p):
            dfood = np.linalg.norm(self.food_pos - p[:, None, :], axis=2)
            return (self._dist_obstacles_ok(p, r + 2 * c["creature_radius"] + 0.05)
                    & (np.linalg.norm(p - self.pos, axis=1) > r + c["creature_radius"] + 0.5)
                    & np.all(~self.food_on | (dfood > r[:, None] + c["food_radius"]), 1))
        if at is None:
            pts, ok = self._sample_point(mask, r + 2 * c["creature_radius"] + 0.05, clear)
        else:
            pts = np.broadcast_to(np.asarray(at, np.float64), (self.E, 2)).copy()
            ok = mask & clear(pts)
        ok &= mask
        slot = np.argmin(self.obs_on, 1)
        idx = np.nonzero(ok)[0]
        self.obs_pos[idx, slot[idx]] = pts[idx]; self.obs_r[idx, slot[idx]] = r[idx]; self.obs_on[idx, slot[idx]] = True
        return ok

    def move_food(self, mask, i, xy):
        """Visitor seed placement: same validity rule as the sampler. Returns accepted mask."""
        c = self.c
        p = np.broadcast_to(np.asarray(xy, np.float64), (self.E, 2)).copy()
        m = 0.4
        ok = (mask & self.food_on[:, i] & (p[:, 0] >= m) & (p[:, 1] >= m) & (p[:, 0] <= c["width"] - m)
              & (p[:, 1] <= c["height"] - m) & self._dist_obstacles_ok(p, c["food_radius"] + 0.1)
              & self._not_in_vents(p) & (np.linalg.norm(p - self.pos, axis=1) > 1.0))
        self.food_pos[ok, i] = p[ok]
        return ok

    def wake_vent(self, mask, v):
        m = mask & self.vent_present[:, v]
        self.vent_on[m, v] = True
        self.vent_timer[m, v] = 300

    def _perturb(self):
        f = self.f
        all_ = np.ones(self.E, bool)
        u = self._u(all_)
        mf = u < f["p_food"]
        mo = (~mf) & (u < f["p_food"] + f["p_obs"])
        mv = (~mf) & (~mo) & (u < f["p_food"] + f["p_obs"] + f["p_vent"])
        if mf.any():
            k = np.minimum(np.floor(self._u(mf) * f["food_n"]).astype(np.int64), f["food_n"] - 1)
            for i in range(f["food_n"]):
                mi = mf & (k == i)
                if mi.any():
                    self._place_food(mi, i)
        if mo.any():
            self.insert_obstacle(mo)
        if mv.any() and f["n_vents"] > 0:
            k = np.minimum(np.floor(self._u(mv) * f["n_vents"]).astype(np.int64), f["n_vents"] - 1)
            for v in range(f["n_vents"]):
                self.wake_vent(mv & (k == v), v)

    # ----------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        c = self.c
        a = np.asarray(action, np.int64)
        rc = c["creature_radius"]
        dturn = np.select([a == 1, a == 2, a == 3, a == 4], [-c["turn_fwd"], c["turn_fwd"], -c["turn_inplace"], c["turn_inplace"]], 0.0)
        spd = np.select([a == 0, a == 1, a == 2], [c["speed"], c["speed_turn_fwd"], c["speed_turn_fwd"]], 0.0)
        self.heading = np.mod(self.heading + dturn, 2 * math.pi)
        old = self.pos.copy()
        p = self.pos + spd[:, None] * np.stack([np.cos(self.heading), np.sin(self.heading)], 1)
        contact = np.zeros(self.E, bool)
        for j in range(c["max_obstacles"]):
            delta = p - self.obs_pos[:, j]
            d = np.linalg.norm(delta, axis=1)
            minD = rc + self.obs_r[:, j]
            hit = self.obs_on[:, j] & (d < minD)
            if hit.any():
                safe = np.where(d > 1e-12, d, 1.0)
                nx = np.where(d > 1e-12, delta[:, 0] / safe, 1.0); ny = np.where(d > 1e-12, delta[:, 1] / safe, 0.0)
                p = np.where(hit[:, None], self.obs_pos[:, j] + np.stack([nx, ny], 1) * minD[:, None], p)
                contact |= hit
        q = np.stack([np.clip(p[:, 0], rc, c["width"] - rc), np.clip(p[:, 1], rc, c["height"] - rc)], 1)
        contact |= np.any(q != p, 1)
        self.pos = q
        moved = np.linalg.norm(q - old, axis=1)
        self.ep_dist += moved
        self.contact = contact; self.ep_contact += contact
        reward = np.zeros(self.E)
        self.energy -= c["energy_decay"] + c["move_cost"] * (spd > 0)
        dv = np.linalg.norm(self.vent_pos - self.pos[:, None, :], axis=2)
        inh = np.any(self.vent_on & (dv < self.vent_r), 1)
        self.in_hazard = inh
        self.energy -= c["hazard_drain"] * inh
        reward += c["reward_hazard_tick"] * inh
        self.ep_hazard += inh
        all_ = np.ones(self.E, bool)
        for i in range(c["max_food"]):
            d = np.linalg.norm(self.food_pos[:, i] - self.pos, axis=1)
            eat = self.food_on[:, i] & (d < rc + c["food_radius"])
            if eat.any():
                self.energy = np.where(eat, np.minimum(1.0, self.energy + c["food_gain"]), self.energy)
                reward += c["reward_food"] * eat
                self.ep_food += eat
                for e in np.nonzero(eat)[0][:64]:
                    self.events.append((int(e), int(self.t[e]), "eat", i))
                self._place_food(eat, i)
        # vent lifecycle
        for v in range(c["max_vents"]):
            pv = self.vent_present[:, v]
            if not pv.any():
                continue
            self.vent_timer[pv, v] -= 1
            flip = pv & (self.vent_timer[:, v] <= 0)
            if flip.any():
                self.vent_on[flip, v] = ~self.vent_on[flip, v]
                self.vent_timer[flip, v] = self._vent_duration(flip, self.vent_on[:, v])[flip]
        self._perturb()
        self.t += 1
        self.last_action = a
        dead = self.energy <= 0
        self.energy = np.maximum(self.energy, 0.0)
        reward += c["reward_death"] * dead
        trunc = (~dead) & (self.t >= c["max_ticks"])
        self.ep_return += reward
        return reward, dead, trunc

    # ----------------------------------------------------------------- observation
    def observe(self) -> np.ndarray:
        c = self.c
        R, rng_ = c["n_rays"], c["ray_range"]
        ang = self.heading[:, None] + self.ray_rel[None, :]
        dx, dy = np.cos(ang), np.sin(ang)  # (E,R)
        px, py = self.pos[:, 0:1], self.pos[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 1e-12, (c["width"] - px) / dx, np.where(dx < -1e-12, (0.0 - px) / dx, np.inf))
            ty = np.where(dy > 1e-12, (c["height"] - py) / dy, np.where(dy < -1e-12, (0.0 - py) / dy, np.inf))
        t_wall = np.minimum(tx, ty)

        def circles(cpos, cr, con):
            ox = cpos[:, None, :, 0] - px[:, :, None]  # (E,1,C)
            oy = cpos[:, None, :, 1] - py[:, :, None]
            proj = ox * dx[:, :, None] + oy * dy[:, :, None]  # (E,R,C)
            d2 = ox * ox + oy * oy
            perp2 = d2 - proj * proj
            r2 = (cr * cr)[:, None, :]
            hit = con[:, None, :] & (perp2 <= r2)
            thc = np.sqrt(np.maximum(r2 - perp2, 0.0))
            t0 = proj - thc; t1 = proj + thc
            t = np.where(t0 >= 0, t0, np.where(t1 >= 0, 0.0, np.inf))
            t = np.where(hit, t, np.inf)
            return t.min(axis=2)

        t_solid = np.minimum(t_wall, circles(self.obs_pos, self.obs_r, self.obs_on))
        fr = np.full(self.food_pos.shape[:2], c["food_radius"])
        t_food = circles(self.food_pos, fr, self.food_on)
        t_haz = circles(self.vent_pos, self.vent_r, self.vent_on)
        prox = lambda t: 1.0 - np.minimum(t, rng_) / rng_
        rays = np.stack([prox(t_solid), prox(t_food), prox(t_haz)], 2).reshape(self.E, R * 3)
        onehot = np.zeros((self.E, N_ACTIONS)); onehot[np.arange(self.E), self.last_action] = 1.0
        return np.concatenate([rays, self.energy[:, None], self.in_hazard[:, None].astype(np.float64),
                               self.contact[:, None].astype(np.float64), onehot], 1)


class VecEnv:
    """Auto-resetting batch env. Episode seeds come from a numpy Generator (non-world randomness)."""

    def __init__(self, n_envs, family="train", seed=0, seed_lo=0, seed_hi=None):
        self.w = World(n_envs, family)
        self.n = n_envs
        self.seed_rng = np.random.default_rng(seed)
        self.seed_lo = seed_lo
        self.seed_hi = seed_hi if seed_hi is not None else C["train_seed_max"]
        self.w.reset(np.ones(n_envs, bool), self._seeds())
        self.w.events.clear()

    def _seeds(self):
        return self.seed_rng.integers(self.seed_lo, self.seed_hi, self.n, dtype=np.int64)

    def obs(self):
        return self.w.observe()

    def step(self, action):
        r, dead, trunc = self.w.step(action)
        done = dead | trunc
        info = None
        if done.any():
            w = self.w
            info = {"food": w.ep_food[done].copy(), "ticks": w.t[done].copy(), "ret": w.ep_return[done].copy(),
                    "dead": dead[done].copy(), "hazard": w.ep_hazard[done].copy()}
            w.reset(done, self._seeds())
        self.w.events.clear()
        return self.obs(), r, dead, trunc, info
