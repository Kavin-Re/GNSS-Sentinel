import copy, math, os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_sentinel.synth import generate
from gnss_sentinel.detectors import (CN0AgcDetector, UniformityDetector, ClockDetector, RaimDetector, InnovationDetector,
                                     CusumDetector, _wh_z, DETECTORS)
from gnss_sentinel.pipeline import Pipeline, latch_series, load_thresholds


def run(det, eps):
    return np.array([det.update(e) for e in eps])


class DetectorTests(unittest.TestCase):
    def test_zero_during_warmup_and_finite(self):
        eps, _ = generate("clean", seed=9001, n=200)
        for D in DETECTORS:
            s = run(D(), eps)
            self.assertTrue(np.all(s[:60] == 0)); self.assertTrue(np.all(np.isfinite(s)), D.name)

    def test_wilson_hilferty_matches_chi2_quantile(self):
        self.assertAlmostEqual(_wh_z(9.488, 4), 1.645, delta=0.12)     # chi2(4) 95th percentile
        self.assertAlmostEqual(_wh_z(3.841, 1), 1.645, delta=0.4)      # coarse for dof=1
        self.assertAlmostEqual(_wh_z(4.0, 4), 0.236, delta=0.02)   # chi2 mean (not median) -> +0.236 in the WH normal approx

    def test_kalman_matches_matrix_form(self):
        rng = np.random.default_rng(0); det = InnovationDetector(); eps, _ = generate("clean", seed=9002, n=120)
        for e in eps: e.pos = e.pos + rng.normal(0, 1, 3)
        # reference: joint 2-state KF per axis, same model/params
        sp, sv, q = det.SP, det.SV, det.Q
        x = [np.array([eps[0].pos[a], eps[0].vel[a]]) for a in range(3)]
        P = [np.diag([sp ** 2, sv ** 2]) for _ in range(3)]
        det._step(eps[0])
        for e in eps[1:60]:
            ref = 0.0
            for a in range(3):
                F = np.array([[1, 1.0], [0, 1]]); Q = q * q * np.array([[1 / 3, .5], [.5, 1]])
                x[a] = F @ x[a]; P[a] = F @ P[a] @ F.T + Q
                H = np.eye(2); R = np.diag([sp ** 2, sv ** 2]); nu = np.array([e.pos[a], e.vel[a]]) - H @ x[a]
                S = H @ P[a] @ H.T + R; ref += nu @ np.linalg.solve(S, nu)
                K = P[a] @ H.T @ np.linalg.inv(S); x[a] = x[a] + K @ nu; P[a] = (np.eye(2) - K @ H) @ P[a]
            got = det._step(e)
            self.assertAlmostEqual(got[0], ref / 6, places=6)

    def test_clock_step_detected_exactly_once(self):
        eps, _ = generate("clean", seed=9003, n=300)
        for e in eps[200:]: e.clock_m += 150.0
        s = run(ClockDetector(), eps)
        self.assertGreater(s[200], 20); self.assertLess(s[:200].max(), 6); self.assertLess(s[201:].max(), 6)

    def test_raim_flags_single_satellite_fault(self):
        eps, m = generate("sat_fault", seed=9004, n=400, onset=200, sat_fault_m=80.0)
        s = run(RaimDetector(), eps)
        self.assertGreater(np.median(s[210:]), 5); self.assertLess(np.percentile(s[60:200], 99), 4)
        d = RaimDetector(); [d.update(e) for e in eps]; self.assertIn(str(m["fault_prn"]), d.explain())

    def test_innovation_flags_position_jump(self):
        eps, _ = generate("clean", seed=9005, n=300)
        for e in eps[200:]: e.pos = e.pos + np.array([60.0, 0, 0])
        s = run(InnovationDetector(), eps)
        self.assertGreater(s[200:206].max(), 10); self.assertLess(s[60:200].max(), 6)

    def test_cusum_arithmetic(self):
        # constant-velocity track with exact velocity, then a drift of 1 m/epoch on E: S grows by ~ (1/sd - k) per epoch
        from gnss_sentinel.epoch import Epoch
        rng = np.random.default_rng(1); det = CusumDetector(warm=60)
        def ep(t, drift):
            e = Epoch(t=float(t)); e.pos = np.array([5.0 * t + drift, 3.0 * t, 0.0]) + rng.normal(0, 0.5, 3)
            e.vel = np.array([5.0, 3.0, 0.0]); return e
        s = [det.update(ep(t, 0.0)) for t in range(100)]
        self.assertLess(max(s), 6)      # clean random walk of the CUSUM stays well under the calibrated threshold (~9)
        sd = det.sd[0]
        s2 = [det.update(ep(100 + i, 1.0 * (i + 1))) for i in range(10)]
        self.assertAlmostEqual(s2[-1] - s2[-6], 5 * (1.0 / sd - det.K), delta=1.5)

    def test_cusum_disabled_without_velocity(self):
        eps, _ = generate("drag", seed=9006, n=500)
        for e in eps: e.vel = np.full(3, np.nan)
        s = run(CusumDetector(), eps); self.assertEqual(s.max(), 0.0)

    def test_latch_hysteresis(self):
        a = latch_series([0, 2, 0.9, 2, 0.5, 0.5, 0.5, 0.5, 2], 1.0)
        self.assertEqual(a.tolist(), [0, 1, 1, 1, 1, 1, 0, 0, 1])

    def test_pipeline_events_have_explanations(self):
        eps, _ = generate("spoof_uniform", seed=9007, n=400, onset=200)
        pl = Pipeline(); ev = []
        for e in eps: ev += pl.update(e)[2]
        starts = [a for a in ev if a.kind == "start"]
        self.assertTrue(starts and all(a.msg for a in starts))
        self.assertTrue(any(a.detector == "uniformity" and a.t >= 200 for a in starts))

    def test_missing_fields_do_not_crash(self):
        eps, _ = generate("clean", seed=9008, n=150)
        for e in eps:
            e.agc_db = e.noise_db = e.clock_m = float("nan"); e.res[:] = np.nan; e.vel[:] = np.nan
        pl = Pipeline()
        for e in eps: pl.update(e)


if __name__ == "__main__":
    unittest.main()
