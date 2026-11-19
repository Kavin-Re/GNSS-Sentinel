"""Passive detectors operating on receiver measurements (Epoch records). Nothing here transmits or generates signals.

Common design
  * The first `warm` epochs (default 60) are used to learn a per-log baseline and ARE ASSUMED CLEAN. A log that
    starts under attack will silently poison its own baseline (documented limit).
  * Each detector returns a continuous score (larger = more anomalous). The score is *debounced*: it is the minimum
    of the last `debounce` raw scores, so `alert == score > threshold` exactly (matters for ROC).
  * Thresholds are calibrated on clean synthetic data (evaluate.py) and shipped in default_thresholds.json.
  * `explain()` returns a human-readable reason for the last scored epoch."""
import math, warnings
from collections import deque
import numpy as np

DET_ORDER = ["cn0_agc", "uniformity", "clock", "raim", "innovation", "cusum"]


class Result:
    __slots__ = ("score", "alert", "msg")

    def __init__(self, score, alert, msg=""):
        self.score, self.alert, self.msg = score, alert, msg


class Detector:
    name = "base"
    debounce = 1

    def __init__(self, warm=60):
        self.warm = warm
        self.n = 0
        self._hist = deque(maxlen=self.debounce)
        self.last = {}
        self.ctx = {}
        self.status = "learning"

    def update(self, ep):
        """Feed one epoch; return the debounced score (0.0 during warm-up)."""
        self.n += 1
        if self.n <= self.warm:
            self._learn(ep)
            if self.n == self.warm:
                self._finish(); self.status = "ok"
            return 0.0
        raw = float(self._score(ep))
        if not math.isfinite(raw): raw = 0.0
        self._hist.append(raw)
        return min(self._hist) if len(self._hist) == self.debounce else 0.0

    def explain(self):
        return ""

    # subclasses
    def _learn(self, ep): pass
    def _finish(self): pass
    def _score(self, ep): return 0.0


def _fin(x):
    return x[np.isfinite(x)]


class CN0AgcDetector(Detector):
    """C/N0 distribution + AGC / noise-floor rule (jamming indicator).
    Per-PRN baseline C/N0 is learned in warm-up. Features per epoch: median and mean-of-worst-3 C/N0 drop relative
    to each satellite's own baseline (high-elevation satellites that disappeared count as a drop to ~18 dB-Hz),
    AGC rise and noise-floor rise. Score = max of the standardised features."""
    name = "cn0_agc"; debounce = 2

    def __init__(self, warm=60):
        super().__init__(warm); self._buf = []; self.base = {}

    def _learn(self, ep): self._buf.append(ep)

    def _finish(self):
        acc = {}
        for ep in self._buf:
            for p, c, e in zip(ep.prn, ep.cn0, ep.el):
                if math.isfinite(c): acc.setdefault(int(p), []).append((c, e))
        need = max(10, 0.5 * len(self._buf))
        self.base = {p: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
                     for p, v in acc.items() if len(v) >= need}
        F = [f for f in (self._feat(ep) for ep in self._buf) if f]
        F = np.array(F) if F else np.zeros((1, 4))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore"); self.mu = np.nanmean(F, 0); self.sd = np.nanstd(F, 0)
        self.sd = np.maximum(np.nan_to_num(self.sd), [0.5, 0.8, 0.25, 0.25])
        self.mu = np.nan_to_num(self.mu)
        self._buf = []

    def _feat(self, ep):
        cur = {int(p): c for p, c in zip(ep.prn, ep.cn0) if math.isfinite(c)}
        drops = []
        for p, (b, e) in self.base.items():
            c = cur.get(p)
            if c is None:
                if e > 25: drops.append(max(b - 18.0, 0.0))
            else:
                drops.append(b - c)
        if len(drops) < 4: return None
        d = np.sort(drops)
        return [float(np.median(d)), float(d[-3:].mean()), ep.agc_db, ep.noise_db]

    def _score(self, ep):
        f = self._feat(ep)
        if f is None:
            self.last = {}; self.ctx = {}
            return 0.0
        z = [(f[i] - self.mu[i]) / self.sd[i] if math.isfinite(f[i]) else -9 for i in range(4)]
        self.last = dict(med=f[0], top=f[1], agc=(f[2] - self.mu[2]) if math.isfinite(f[2]) else 0.0,
                         noise=(f[3] - self.mu[3]) if math.isfinite(f[3]) else 0.0)
        self.ctx = dict(f=f, z=z, n=ep.n_sat)
        return max(z)

    def explain(self):
        if not self.ctx: return "too few baseline satellites tracked"
        f, z = self.ctx["f"], self.ctx["z"]
        parts = []
        if z[2] > 3: parts.append("AGC %+.1f dB vs baseline" % (f[2] - self.mu[2]))
        if z[3] > 3: parts.append("noise floor %+.1f dB" % (f[3] - self.mu[3]))
        parts.append("median C/N0 drop %.1f dB, worst-3 mean drop %.1f dB" % (f[0] - self.mu[0], f[1] - self.mu[1]))
        wide = z[0] > 3 and (z[2] > 3 or z[3] > 3)
        narrow = z[1] > 3 and z[0] < 3
        added = (z[2] > 3 or z[3] > 3) and z[0] < 1 and z[1] < 3
        kind = ("wideband-jamming-like (all satellites degrade with AGC/noise rise)" if wide else
                "narrowband/CW-like (a few satellites hit hard, rest nearly unaffected)" if narrow else
                "in-band power rise without C/N0 loss (added-signal-like, e.g. repeater/spoofer power; not jamming-like)" if added else
                "interference-like C/N0 degradation")
        return "; ".join(parts) + " -> " + kind


class UniformityDetector(Detector):
    """All-satellite power-uniformity test (spoofing indicator): genuine signals show a C/N0 spread of several dB
    (elevation, antenna pattern, obstruction); a single-source spoofer tends to make all satellites nearly equal.
    Score = (baseline_spread - median(last 3 spreads)) / baseline_sd. Suppressed when mean C/N0 has collapsed
    (that is the jamming regime, handled by cn0_agc)."""
    name = "uniformity"; debounce = 2

    def __init__(self, warm=60):
        super().__init__(warm); self._s = []; self._m = []; self._w = deque(maxlen=3)

    def _stat(self, ep):
        c = _fin(ep.cn0)
        if len(c) < 6: return None
        return float(np.std(c, ddof=1)), float(c.mean())

    def _learn(self, ep):
        s = self._stat(ep)
        if s: self._s.append(s[0]); self._m.append(s[1])

    def _finish(self):
        s = np.array(self._s or [3.0]); self.mu = float(s.mean()); self.sd = max(float(s.std()), 0.2)
        self.mean_c = float(np.mean(self._m or [40.0]))

    def _score(self, ep):
        s = self._stat(ep)
        if not s: return 0.0
        self._w.append(s[0]); sp = float(np.median(self._w))
        self.last = dict(spread=sp, meandrop=self.mean_c - s[1])
        self.ctx = dict(sp=sp, mean=s[1])
        if s[1] < self.mean_c - 3: return 0.0
        return (self.mu - sp) / self.sd

    def explain(self):
        c = self.ctx
        return ("C/N0 spread across satellites %.2f dB-Hz vs baseline %.2f +/- %.2f; mean C/N0 %.1f (baseline %.1f) "
                "-> all satellites near-equal power, typical of a single-source (spoofer-like) signal"
                % (c["sp"], self.mu, self.sd, c["mean"], self.mean_c))


class ClockDetector(Detector):
    """Clock-bias jump test. e = (clock_t - clock_{t-1}) - median(last 10 increments); score = |e| / robust sigma
    (sigma = 1.4826*MAD learned in warm-up). Meaconing/replay with a common delay shows up here as a step."""
    name = "clock"; debounce = 1

    def __init__(self, warm=60):
        super().__init__(warm); self._prev = None; self._inc = deque(maxlen=10); self._e = []

    def _resid(self, ep):
        c = ep.clock_m
        if not math.isfinite(c): self._prev = None; return None
        r = None
        if self._prev is not None:
            inc = c - self._prev
            if len(self._inc) >= 5: r = inc - float(np.median(self._inc))
            self._inc.append(inc)
        self._prev = c
        return r

    def _learn(self, ep):
        r = self._resid(ep)
        if r is not None: self._e.append(r)

    def _finish(self):
        e = np.array(self._e or [1.0])
        self.sd = max(1.4826 * float(np.median(np.abs(e - np.median(e)))), 0.3)

    def _score(self, ep):
        r = self._resid(ep)
        if r is None: return 0.0
        self.ctx = dict(r=r); self.last = dict(z=abs(r) / self.sd)
        return abs(r) / self.sd

    def explain(self):
        r = self.ctx["r"]
        return ("clock-bias increment deviates by %+.1f m (= %+.3f us, %.1f sigma) from the recent drift; "
                "a common step on all satellites is the signature of a replay/meaconing delay or receiver clock event"
                % (r, r / 299.792458, abs(r) / self.sd))


def _wh_z(T, dof):
    """Wilson-Hilferty normal approximation of a chi-square(dof) variate."""
    a = 2.0 / (9 * dof)
    return ((T / dof) ** (1 / 3) - (1 - a)) / math.sqrt(a)


class RaimDetector(Detector):
    """RAIM-style residual consistency: weighted sum of squared post-fit pseudorange residuals, chi-square with
    n-4 dof (needs >= 5 satellites). Scale learned in warm-up. Catches inconsistent measurement sets (multipath burst,
    per-satellite offsets, noisy jammed tracking); a *self-consistent* spoofer is invisible to it by construction."""
    name = "raim"; debounce = 2

    def __init__(self, warm=60):
        super().__init__(warm); self._r = []

    @staticmethod
    def _sse(ep):
        r = ep.res; ok = np.isfinite(r)
        n = int(ok.sum())
        if n < 5 or not ep.has_fix: return None
        w = np.maximum(np.sin(np.radians(ep.el[ok])), 0.35)
        return float(((r[ok] * w) ** 2).sum()), n - 4, ok, w

    def _learn(self, ep):
        s = self._sse(ep)
        if s: self._r.append(s[0] / s[1])

    def _finish(self):
        self.k = max(float(np.mean(self._r)) if self._r else 1.0, 1e-3)

    def _score(self, ep):
        s = self._sse(ep)
        if not s: return 0.0
        sse, dof, ok, w = s
        z = _wh_z(sse / self.k, dof)
        j = int(np.argmax(np.abs(ep.res[ok]) * w))
        self.ctx = dict(T=sse / self.k, dof=dof, z=z, prn=int(ep.prn[ok][j]), rn=float(abs(ep.res[ok][j]) * w[j] / math.sqrt(self.k)))
        self.last = dict(z=z)
        return z

    def explain(self):
        c = self.ctx
        return ("residual test statistic %.1f vs chi-square(%d) expectation %d (z=%.1f); largest weighted residual on PRN %d "
                "(%.1f sigma) -> measurements mutually inconsistent" % (c["T"], c["dof"], c["dof"], c["z"], c["prn"], c["rn"]))


class InnovationDetector(Detector):
    """Innovation monitor: per-axis constant-velocity Kalman filter fusing receiver position (sigma_p) and
    receiver velocity (sigma_v); normalised innovation squared (NIS) averaged over a 5-epoch window and standardised
    with warm-up statistics. Missing axes/measurements (NaN) are skipped, so position-only logs still work."""
    name = "innovation"; debounce = 1
    SP, SV, Q = 2.0, 0.1, 0.6

    def __init__(self, warm=60, win=5):
        super().__init__(warm)
        self.x = None; self.P = None; self.tprev = None
        self.win = deque(maxlen=win); self._r = []; self.W = win

    def _step(self, ep):
        if not ep.has_fix: return None
        if self.x is None:
            self.x = [[ep.pos[a], ep.vel[a] if math.isfinite(ep.vel[a]) else 0.0] for a in range(3)]
            self.P = [[[self.SP ** 2, 0.0], [0.0, self.SV ** 2 if math.isfinite(ep.vel[a]) else 25.0]] for a in range(3)]
            self.tprev = ep.t
            return None
        dt = max(ep.t - self.tprev, 1e-3); self.tprev = ep.t
        nis, m, big = 0.0, 0, (0.0, "")
        q = self.Q ** 2
        for a in range(3):
            x, P = self.x[a], self.P[a]
            # predict
            p, v = x[0] + dt * x[1], x[1]
            P00 = P[0][0] + dt * (P[0][1] + P[1][0]) + dt * dt * P[1][1] + q * dt ** 3 / 3
            P01 = P[0][1] + dt * P[1][1] + q * dt ** 2 / 2
            P11 = P[1][1] + q * dt
            for (mi, z, R) in ((0, ep.pos[a], self.SP ** 2), (1, ep.vel[a], self.SV ** 2)):
                if not math.isfinite(z): continue
                if mi == 0:
                    S = P00 + R; nu = z - p; K0, K1 = P00 / S, P01 / S
                    p += K0 * nu; v += K1 * nu
                    P11 -= K1 * P01; P01 -= K0 * P01; P00 -= K0 * P00
                else:
                    S = P11 + R; nu = z - v; K0, K1 = P01 / S, P11 / S
                    p += K0 * nu; v += K1 * nu
                    P00 -= K0 * P01; P01 -= K1 * P01; P11 -= K1 * P11
                u = nu * nu / S; nis += u; m += 1
                if u > big[0]: big = (u, "%s %s innovation %+.1f %s (%.1f sigma)" % ("ENU"[a], "position" if mi == 0 else "velocity", nu, "m" if mi == 0 else "m/s", math.sqrt(u)))
            x[0], x[1] = p, v; P[0][0], P[0][1], P[1][0], P[1][1] = P00, P01, P01, P11
        return nis / m, m, big[1]

    def _learn(self, ep):
        r = self._step(ep)
        if r: self._r.append(r[0])

    def _finish(self):
        r = np.array(self._r or [1.0]); self.mu = float(r.mean()); self.sd = max(float(r.std()), 0.4)

    def _score(self, ep):
        r = self._step(ep)
        if r is None: return 0.0
        self.win.append(r[0])
        if len(self.win) < self.W: return 0.0
        mean = float(np.mean(self.win)); z = (mean - self.mu) / (self.sd / math.sqrt(self.W))
        self.ctx = dict(mean=mean, worst=r[2]); self.last = dict(z=z)
        return z

    def explain(self):
        c = self.ctx
        return ("normalised innovation squared %.1fx baseline over last %d epochs; worst now: %s -> receiver position/velocity "
                "inconsistent with the motion model" % (c["mean"] / max(self.mu, 1e-6), self.W, c["worst"]))


class CusumDetector(Detector):
    """Page/CUSUM test for slow position drag. d = (p_t - p_{t-1}) - 0.5 (v_t + v_{t-1}) dt, i.e. how far the position
    moved beyond what the receiver's own velocity (Doppler-derived) explains. Per-axis two-sided CUSUM on d/sigma_d
    with drift k; score = largest CUSUM statistic. Assumes the spoofer does not also shift the Doppler-derived velocity
    consistently (a naive drag). Needs velocity: disabled (score 0) on position-only logs."""
    name = "cusum"; debounce = 1
    K = 0.1

    def __init__(self, warm=60):
        super().__init__(warm); self.prev = None; self._d = []; self.recent = deque(maxlen=30)
        self.Sp = np.zeros(3); self.Sm = np.zeros(3); self.sd = np.ones(3); 

    def _d_of(self, ep):
        if not ep.has_fix or not np.isfinite(ep.vel).any(): self.prev = None; return None
        cur = (ep.t, ep.pos.copy(), ep.vel.copy()); prev, self.prev = self.prev, cur
        dt = cur[0] - prev[0] if prev is not None else None
        if dt is None or not (0 < dt <= 3): return None      # gap, duplicate or out-of-order timestamp: skip this pair
        return (cur[1] - prev[1]) - 0.5 * (cur[2] + prev[2]) * dt

    def _learn(self, ep):
        d = self._d_of(ep)
        if d is not None: self._d.append(d)

    def _finish(self):
        D = np.array(self._d) if self._d else np.ones((2, 3))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore"); sd = np.nanstd(D, 0)      # an all-NaN axis (e.g. NMEA has no vertical velocity) is expected
        self.sd = np.maximum(np.nan_to_num(sd, nan=1.0), 0.5)

    def _score(self, ep):
        d = self._d_of(ep)
        if d is None: return 0.0
        z = np.nan_to_num(d / self.sd); self.recent.append(np.nan_to_num(d))
        self.Sp = np.maximum(0, self.Sp + z - self.K); self.Sm = np.maximum(0, self.Sm - z - self.K)
        # bookkeeping for a drift-rate estimate: reset when both sides are zero
        S = np.maximum(self.Sp, self.Sm); a = int(np.argmax(S))
        self.ctx = dict(a=a, sign=1 if self.Sp[a] >= self.Sm[a] else -1, S=float(S[a]), n=len(self.recent),
                        rate=float(np.mean(self.recent, 0)[a]))
        self.last = dict(S=float(S[a]))
        return float(S[a])

    def explain(self):
        c = self.ctx
        return ("position drifting beyond receiver velocity on %s axis (%s): CUSUM %.1f, ~%.2f m/s over the last %d epochs (recent mean excess) -> "
                "slow position drag / spoofer pull-off" % ("ENU"[c["a"]], "+" if c["sign"] > 0 else "-", c["S"], abs(c["rate"]), c["n"]))


DETECTORS = [CN0AgcDetector, UniformityDetector, ClockDetector, RaimDetector, InnovationDetector, CusumDetector]
