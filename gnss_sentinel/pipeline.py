"""Streaming pipeline: runs all detectors epoch by epoch, tracks alert start/end events, exposes classifier features."""
import json, os
from collections import deque
from dataclasses import dataclass
import numpy as np
from .detectors import DETECTORS, DET_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_NAMES = ["cn0_med_drop", "cn0_top3_drop", "agc_rise", "noise_rise", "cn0_spread", "cn0_mean_drop", "cn0_el_corr",
             "mp_std", "clock_z", "raim_z", "innov_z", "cusum_S", "n_sat"]
WINDOW = 30


def load_thresholds(path=None):
    with open(path or os.path.join(HERE, "default_thresholds.json")) as f:
        return json.load(f)["thresholds"]


@dataclass
class Alert:
    t: float
    detector: str
    kind: str          # "start" | "end"
    score: float
    thr: float
    msg: str = ""


def _epoch_raw(ep, dets):
    c = ep.cn0[np.isfinite(ep.cn0)]
    el = ep.el[np.isfinite(ep.cn0)]
    corr = 0.0
    if len(c) >= 6 and c.std() > 1e-9:
        s = np.sin(np.radians(el))
        if s.std() > 1e-9: corr = float(np.corrcoef(c, s)[0, 1])
    mp = ep.mp[np.isfinite(ep.mp)]
    g = lambda d, k: float(dets[d].last.get(k, 0.0)) if dets[d].last else 0.0
    return [g("cn0_agc", "med"), g("cn0_agc", "top"), g("cn0_agc", "agc"), g("cn0_agc", "noise"),
            g("uniformity", "spread"), g("uniformity", "meandrop"), corr, float(mp.std()) if len(mp) > 3 else 0.0,
            g("clock", "z"), g("raim", "z"), g("innovation", "z"), g("cusum", "S"), float(ep.n_sat)]


CLEAR_RATIO, CLEAR_HOLD = 0.6, 3


class AlertLatch:
    """Hysteresis: start when score > thr; end when score < CLEAR_RATIO*thr for CLEAR_HOLD consecutive epochs."""
    def __init__(self, thr):
        self.thr = thr; self.on = False; self.low = 0

    def step(self, s):
        if not self.on:
            if s > self.thr: self.on = True; self.low = 0
        else:
            self.low = self.low + 1 if s < CLEAR_RATIO * self.thr else 0
            if self.low >= CLEAR_HOLD: self.on = False
        return self.on


def latch_series(scores, thr):
    """Boolean array of alert state with hysteresis for a whole score series (same logic as the streaming pipeline)."""
    L = AlertLatch(thr); return np.array([L.step(float(s)) for s in scores], bool)


class Pipeline:
    def __init__(self, thresholds=None, warm=60):
        self.thr = thresholds or load_thresholds()
        self.dets = {d.name: d(warm=warm) for d in DETECTORS}
        self.active = {n: False for n in DET_ORDER}
        self.latch = {n: AlertLatch(self.thr[n]) for n in DET_ORDER}
        self.scores_hist = deque(maxlen=WINDOW); self.raw_hist = deque(maxlen=WINDOW)
        self.warm = warm; self.n = 0

    def update(self, ep, explain=True):
        """Returns (scores: list in DET_ORDER, raw: list, events: list[Alert])."""
        self.n += 1
        scores, events = [], []
        for name in DET_ORDER:
            d = self.dets[name]
            s = d.update(ep); scores.append(s)
            on = self.latch[name].step(s)
            if on != self.active[name]:
                events.append(Alert(ep.t, name, "start" if on else "end", s, self.thr[name],
                                    d.explain() if (on and explain) else ""))
                self.active[name] = on
        raw = _epoch_raw(ep, self.dets) if self.n > self.warm else [0.0] * len(RAW_NAMES)
        self.scores_hist.append(scores); self.raw_hist.append(raw)
        return scores, raw, events

    def window_features(self):
        if self.n < self.warm + WINDOW: return None
        return window_features(np.array(self.scores_hist), np.array(self.raw_hist))


def window_features(S, R):
    """S: (W,6) detector scores, R: (W,len(RAW_NAMES)) -> classifier feature vector (mean and max of each column)."""
    X = np.c_[S, R]
    return np.r_[X.mean(0), X.max(0)]
