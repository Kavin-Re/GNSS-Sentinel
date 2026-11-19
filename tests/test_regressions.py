"""Regression tests for bugs fixed during review, plus a static check of the 'passive, offline' claim."""
import ast, contextlib, io, os, sys, unittest, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_sentinel import cli, logio
from gnss_sentinel.dashboard import Recorder, build_html
from gnss_sentinel.detectors import CN0AgcDetector, CusumDetector
from gnss_sentinel.epoch import Epoch
from gnss_sentinel.pipeline import load_thresholds
from gnss_sentinel.synth import generate

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gnss_sentinel")


class RegressionTests(unittest.TestCase):
    def test_nmea_stream_emits_no_numpy_warning(self):
        # NMEA carries no vertical velocity, so one CUSUM axis is all-NaN in warm-up; that used to leak a RuntimeWarning
        eps, _ = generate("clean", seed=3, n=120); buf = io.StringIO(); logio.write_nmea(buf, eps)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cli.run_stream(buf.getvalue().splitlines(True), out=io.StringIO(), quiet=True, use_clf=False)

    def test_bad_nmea_sentences_are_reported(self):
        eps, _ = generate("clean", seed=3, n=5); buf = io.StringIO(); logio.write_nmea(buf, eps)
        lines = buf.getvalue().splitlines(True)
        lines.insert(3, lines[0].rsplit("*", 1)[0] + "*00\n")          # corrupt checksum
        r = logio.read_log(lines); n = len(list(r))
        self.assertEqual(n, 5); self.assertEqual(r.bad, 1)
        out = io.StringIO(); cli.run_stream(lines, out=out, quiet=True, use_clf=False)
        self.assertIn("1 malformed NMEA sentence", out.getvalue())

    def test_cusum_ignores_duplicate_and_backward_timestamps(self):
        det = CusumDetector(warm=5)
        def ep(t, x):
            e = Epoch(t=float(t)); e.pos = np.array([x, 0.0, 0.0]); e.vel = np.zeros(3); return e
        for t in range(6): det.update(ep(t, 0.0))
        self.assertEqual(det.update(ep(5, 500.0)), 0.0)                # duplicate timestamp: pair skipped
        self.assertEqual(det.update(ep(4, 900.0)), 0.0)                # time going backwards: pair skipped

    def test_cn0_explanation_does_not_call_a_rise_degradation(self):
        eps, _ = generate("clean", seed=5, n=120)
        det = CN0AgcDetector()
        for e in eps[:100]: det.update(e)
        e = eps[100]; e.cn0 = e.cn0 + 4.0; e.agc_db += 3.0             # more in-band power, C/N0 up (meaconing-like)
        det.update(e); msg = det.explain()
        self.assertIn("without C/N0 loss", msg); self.assertNotIn("degradation", msg)

    def test_dashboard_escapes_log_supplied_text(self):
        eps, _ = generate("clean", seed=6, n=70)
        eps[65].label = "</script><script>alert(1)</script><!--"
        rec = Recorder(load_thresholds())
        for e in eps: rec.add(e, [0.0] * 6, [])
        html = build_html(rec, title="<img src=x onerror=alert(1)>")
        self.assertEqual(html.count("<script"), 2); self.assertEqual(html.count("</script>"), 2)
        self.assertNotIn("<img src=x", html); self.assertNotIn("<!--", html)

    def test_cli_reports_bad_input_without_traceback(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["stream", os.path.join(PKG, "does-not-exist.csv")])
        self.assertEqual(rc, 2); self.assertIn("error", err.getvalue())

    def test_package_is_offline_and_receive_only(self):
        """The toolkit must stay a log analyser: no network, serial, SDR or subprocess modules anywhere in the package."""
        banned = {"socket", "serial", "usb", "SoapySDR", "rtlsdr", "hackrf", "uhd", "gnuradio", "bladerf", "subprocess",
                  "requests", "urllib", "http", "asyncio", "ctypes"}
        for fn in sorted(os.listdir(PKG)):
            if not fn.endswith(".py"): continue
            with open(os.path.join(PKG, fn)) as f: tree = ast.parse(f.read(), fn)
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import): mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0: mods = [node.module or ""]
                for m in mods:
                    self.assertNotIn(m.split(".")[0], banned, "%s imports %s" % (fn, m))


if __name__ == "__main__":
    unittest.main()
