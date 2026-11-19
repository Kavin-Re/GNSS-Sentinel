import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_sentinel.synth import generate, SCENARIOS


class SynthTests(unittest.TestCase):
    def test_deterministic(self):
        a, ma = generate("cw", seed=4, n=300); b, mb = generate("cw", seed=4, n=300)
        self.assertEqual(ma, mb)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.cn0, y.cn0); np.testing.assert_array_equal(x.pos, y.pos)

    def test_shapes_and_labels(self):
        for sc in SCENARIOS:
            eps, meta = generate(sc, seed=2, n=400)
            self.assertEqual(len(eps), 400)
            for e in eps[::50]:
                self.assertEqual(len(e.prn), e.n_sat); self.assertEqual(len(e.cn0), e.n_sat)
                self.assertEqual(len(e.res), e.n_sat); self.assertEqual(len(e.mp), e.n_sat)
            on = meta["onset"]
            self.assertTrue(all(e.label == "clean" for e in eps[:on]))
            self.assertTrue(all(e.label == (sc if sc != "clean" else "clean") for e in eps[on:]))

    def test_clean_statistics(self):
        eps, _ = generate("clean", seed=7, n=300)
        self.assertTrue(all(9 <= e.n_sat <= 14 for e in eps))
        sp = np.mean([np.std(e.cn0) for e in eps]); self.assertTrue(1.5 < sp < 4.5, sp)
        self.assertTrue(all(np.all(e.cn0 == np.round(e.cn0)) for e in eps))  # integer dB-Hz like real receivers

    def test_residuals_satisfy_normal_equations(self):
        eps, _ = generate("clean", seed=3, n=80)
        for e in eps[::10]:
            el, az = np.radians(e.el), np.radians(e.az)
            los = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], 1)
            G = np.c_[-los, np.ones(len(el))]; w = np.maximum(np.sin(el), 0.35) ** 2
            np.testing.assert_allclose(G.T @ (w * e.res), 0, atol=0.25)   # weights in the generator also include a small C/N0-rounding term the test cannot see

    def test_drag_is_exact_ramp_paired_with_clean(self):
        c, _ = generate("clean", seed=5, n=500); d, m = generate("drag", seed=5, n=500)
        on, r = m["onset"], m["drag_rate_mps"]
        np.testing.assert_allclose(d[on - 1].pos, c[on - 1].pos, atol=1e-9)
        got = np.linalg.norm(d[499].pos - c[499].pos)
        self.assertAlmostEqual(got, r * (499 - on), places=6)
        np.testing.assert_allclose(d[499].vel, c[499].vel)       # naive drag leaves Doppler velocity untouched (documented assumption)

    def test_meaconing_is_clock_step(self):
        c, _ = generate("clean", seed=6, n=500); a, m = generate("meacon", seed=6, n=500)
        step = a[499].clock_m - c[499].clock_m
        self.assertLess(abs(step - m["clock_jump_m"]), 40)       # per-satellite repeater offsets perturb the fitted clock a little
        self.assertEqual(a[m["onset"] - 1].clock_m, c[m["onset"] - 1].clock_m)

    def test_spoof_uniform_power(self):
        eps, m = generate("spoof_uniform", seed=8, n=500); on = m["onset"]
        pre = np.mean([np.std(e.cn0) for e in eps[:on]]); post = np.mean([np.std(e.cn0) for e in eps[on + 10:]])
        self.assertGreater(pre, 1.8); self.assertLess(post, 0.8)
        self.assertLess(np.mean([np.std(e.mp) for e in eps[on + 10:]]), np.mean([np.std(e.mp) for e in eps[:on]]))

    def test_jam_ramp_degrades_snr_and_raises_agc(self):
        eps, m = generate("jam_ramp", seed=9, n=720, onset=200)
        self.assertGreater(eps[-1].agc_db - eps[100].agc_db, 5)
        self.assertGreater(eps[-1].noise_db - eps[100].noise_db, 5)
        self.assertLess(np.nanmean(eps[-1].cn0) if eps[-1].n_sat else 0, np.nanmean(eps[100].cn0) - 8)

    def test_cw_hits_few_satellites_hard(self):
        eps, m = generate("cw", seed=10, n=500, onset=150); base, post = eps[100], eps[400]
        common = [p for p in base.prn if p in post.prn]
        drops = np.array([base.cn0[list(base.prn).index(p)] - post.cn0[list(post.prn).index(p)] for p in common])
        self.assertGreater(drops.max(), 6); self.assertLess(np.median(drops), drops.max() - 4)

    def test_no_rf_or_iq_api(self):
        import gnss_sentinel.synth as s
        self.assertFalse([n for n in dir(s) if any(k in n.lower() for k in ("iq", "waveform", "transmit", "modulate"))])

    def test_bad_scenario(self):
        with self.assertRaises(ValueError): generate("nope")


if __name__ == "__main__":
    unittest.main()
