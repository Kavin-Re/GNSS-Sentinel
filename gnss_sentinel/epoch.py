"""Receiver-level measurement record used by every module (one epoch = one receiver output interval).

All arrays are per tracked satellite and have equal length. Missing scalars are NaN."""
from dataclasses import dataclass, field
import numpy as np

NAN = float("nan")


def _f(n=0):
    return np.zeros(n, float)


@dataclass
class Epoch:
    t: float                                  # seconds (monotonic, receiver/UTC time)
    n_sat: int = 0                            # satellites used in the solution
    prn: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # satellite id (unique per constellation)
    el: np.ndarray = field(default_factory=_f)    # elevation, deg
    az: np.ndarray = field(default_factory=_f)    # azimuth, deg
    cn0: np.ndarray = field(default_factory=_f)   # C/N0, dB-Hz (NaN = visible but not tracked)
    res: np.ndarray = field(default_factory=_f)   # post-fit pseudorange residual, m
    mp: np.ndarray = field(default_factory=_f)    # multipath metric (code-minus-carrier style), m
    agc_db: float = NAN                       # AGC state, dB relative to nominal (+ = more in-band power)
    noise_db: float = NAN                     # estimated noise-floor rise, dB relative to nominal
    clock_m: float = NAN                      # receiver clock bias, metres (c * dt)
    pos: np.ndarray = field(default_factory=lambda: np.full(3, NAN))   # local ENU position, m
    vel: np.ndarray = field(default_factory=lambda: np.full(3, NAN))   # local ENU velocity, m/s
    label: str = ""                           # ground truth (synthetic logs only; never read by detectors)

    @property
    def has_fix(self):
        return bool(np.isfinite(self.pos).all())
