import io, itertools, os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_sentinel.synth import generate
from gnss_sentinel import logio


class LogIoTests(unittest.TestCase):
    def test_checksum_known_sentence(self):
        self.assertEqual(logio.checksum("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"), "47")

    def test_csv_roundtrip(self):
        eps, _ = generate("cw", seed=1, n=120)
        buf = io.StringIO(); logio.write_csv(buf, eps, header_comment="test"); buf.seek(0)
        back = list(logio.read_csv(buf)); self.assertEqual(len(back), len(eps))
        for a, b in zip(eps, back):
            self.assertEqual(a.n_sat, b.n_sat); np.testing.assert_array_equal(a.prn, b.prn); self.assertEqual(a.label, b.label)
            np.testing.assert_allclose(a.cn0, b.cn0, atol=0.06); np.testing.assert_allclose(a.pos, b.pos, atol=1e-3, equal_nan=True)
            np.testing.assert_allclose(a.res, b.res, atol=1e-3, equal_nan=True); self.assertAlmostEqual(a.clock_m, b.clock_m, 2) if a.has_fix else None

    def test_csv_streams_lazily(self):
        eps, _ = generate("clean", seed=1, n=20)
        buf = io.StringIO(); logio.write_csv(buf, eps); lines = buf.getvalue().splitlines(True)
        pulled = []
        def src():
            for l in lines: pulled.append(1); yield l
        it = logio.read_csv(src()); next(it)
        self.assertEqual(len(pulled), 2)         # header + first row only

    def test_csv_minimal_columns(self):
        txt = "t,n_sat,prn0,cn00\n0,1,5,44\n1,1,5,43\n"
        # column names are prn{k}/cn0{k}: 'cn00' is cn0 of slot 0
        eps = list(logio.read_csv(io.StringIO(txt)))
        self.assertEqual(len(eps), 2); self.assertEqual(eps[0].cn0[0], 44); self.assertTrue(np.isnan(eps[0].agc_db)); self.assertFalse(eps[0].has_fix)

    def test_nmea_roundtrip(self):
        eps, _ = generate("clean", seed=2, n=60)
        buf = io.StringIO(); logio.write_nmea(buf, eps); text = buf.getvalue()
        self.assertIn("$GPGSV,", text); self.assertIn("$GPGGA,", text)
        back = list(logio.read_nmea(io.StringIO(text))); self.assertEqual(len(back), len(eps))
        o = eps[0].pos
        for a, b in zip(eps[:30], back[:30]):
            self.assertEqual(a.n_sat, b.n_sat); self.assertEqual(len(a.prn), len(b.prn))
            np.testing.assert_allclose(np.sort(np.round(a.cn0)), np.sort(b.cn0))
            np.testing.assert_allclose(a.pos - o, b.pos, atol=0.12)   # NMEA ddmm.mmmm with 4 decimals = 0.185 m resolution
            np.testing.assert_allclose(a.vel[:2], b.vel[:2], atol=0.06); self.assertTrue(np.isnan(b.vel[2]))
        self.assertAlmostEqual(back[10].t - back[0].t, 10.0, 3)

    def test_nmea_bad_checksum_and_garbage(self):
        good = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
        bad = good[:-2] + "00"
        r = logio.NmeaReader()
        self.assertEqual(r.feed(bad), []); self.assertEqual(r.bad, 1)
        self.assertEqual(r.feed("garbage"), []); self.assertEqual(r.feed("$PUBX,00*33"), [])
        r.feed(good); out = r.flush(); self.assertEqual(len(out), 1); self.assertEqual(out[0].n_sat, 8)

    def test_nmea_gsv_signal_id_multi_talker_and_missing_snr(self):
        s = lambda b: "$%s*%s" % (b, logio.checksum(b))
        lines = [s("GNGGA,000001.00,1000.0000,N,07800.0000,E,1,09,1.0,100.0,M,0.0,M,,"),
                 s("GPGSV,2,1,05,03,40,100,45,07,20,200,38,09,60,300,,11,10,050,31,1"),     # NMEA 4.10: trailing signal id
                 s("GIGSV,1,1,01,02,30,150,40,1"),                                        # generic talker (e.g. NavIC style) accepted by suffix
                 s("GNGGA,000002.00,1000.0001,N,07800.0000,E,1,09,1.0,100.0,M,0.0,M,,")]
        eps = list(logio.read_nmea(lines)); self.assertEqual(len(eps), 2)
        e = eps[0]; self.assertEqual(len(e.prn), 5)
        self.assertEqual(int(np.isfinite(e.cn0).sum()), 4)                               # PRN 09 has an empty SNR -> not tracked
        self.assertEqual(sorted(e.cn0[np.isfinite(e.cn0)].tolist()), [31, 38, 40, 45])
        self.assertEqual(len(set(int(p) // 1000 for p in e.prn)), 2)                      # talkers kept apart
        self.assertAlmostEqual(eps[1].pos[1], (1 / 600000.0) * 6378137.0 * np.pi / 180, delta=0.01)   # 0.0001 arc-min north, 1st fix is origin

    def test_nmea_rmc_before_gga_and_midnight(self):
        s = lambda b: "$%s*%s" % (b, logio.checksum(b))
        L = [s("GPRMC,235959.00,A,1000.0000,N,07800.0000,E,10.0,90.0,010100,,,A"),
             s("GPGGA,235959.00,1000.0000,N,07800.0000,E,1,08,1.0,100.0,M,0.0,M,,"),
             s("GPRMC,000000.00,A,1000.0000,N,07800.0000,E,10.0,90.0,020100,,,A"),
             s("GPGGA,000000.00,1000.0000,N,07800.0000,E,1,08,1.0,100.0,M,0.0,M,,")]
        eps = list(logio.read_nmea(L)); self.assertEqual(len(eps), 2)
        self.assertAlmostEqual(eps[1].t - eps[0].t, 1.0)
        self.assertAlmostEqual(eps[0].vel[0], 10 * logio.KN, places=3); self.assertAlmostEqual(eps[0].vel[1], 0, places=3)

    def test_auto_detect(self):
        self.assertEqual(logio.detect_format(["# c", "$GPGGA,..."]), "nmea"); self.assertEqual(logio.detect_format(["t,n_sat"]), "csv")
        eps, _ = generate("clean", seed=1, n=5); b = io.StringIO(); logio.write_nmea(b, eps)
        self.assertEqual(len(list(logio.read_log(b.getvalue().splitlines(True)))), 5)


if __name__ == "__main__":
    unittest.main()
