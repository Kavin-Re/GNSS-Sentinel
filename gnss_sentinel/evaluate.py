"""Evaluation on SYNTHETIC data: threshold calibration, ROC/AUC, detection delay, false alarms, ablation, classifier.

Protocol (seed ranges are disjoint):
  calibration  seeds 1000+  clean runs only -> per-detector threshold = 99th percentile of the per-run max score
  classifier   seeds 2000+  training windows (all 6 classes)
  test         seeds 5000+  held out; nominal generator profile
  shifted      seeds 7000+  held out; 'shifted' generator profile (noisier, heavier tails) with nominal thresholds/model
Run-level statistics use windows [onset, onset+WIN) of each run. Everything is synthetic => optimistic."""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from .synth import generate, SCENARIOS, ATTACKS
from .pipeline import Pipeline, latch_series, window_features, RAW_NAMES, WINDOW, HERE
from .detectors import DET_ORDER
from . import classifier as clf

WIN = 420
N_STEPS = 720
Z95 = 1.959964


def simulate(job):
    scenario, seed, profile = job
    eps, meta = generate(scenario, seed=seed, n=N_STEPS, profile=profile)
    dummy = {n: 1e9 for n in DET_ORDER}
    pl = Pipeline(thresholds=dummy)
    S, R = [], []
    for e in eps:
        s, r, _ = pl.update(e, explain=False); S.append(s); R.append(r)
    return dict(scenario=scenario, seed=seed, onset=meta["onset"], S=np.array(S, np.float32), R=np.array(R, np.float32))


def run_jobs(jobs, procs=2):
    if procs > 1:
        with Pool(procs) as p: return p.map(simulate, jobs, chunksize=4)
    return [simulate(j) for j in jobs]


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def poisson_ci(k, exposure, alpha=0.05):
    """Exact (Garwood) CI for a Poisson rate = k/exposure, by bisection on the Poisson cdf."""
    def cdf(kk, lam): return sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(kk + 1)) if lam > 0 else 1.0
    lo = 0.0
    if k > 0:
        a, b = 0.0, k + 10 * math.sqrt(k) + 10
        for _ in range(80):
            m = (a + b) / 2
            if 1 - cdf(k - 1, m) > alpha / 2: b = m
            else: a = m
        lo = (a + b) / 2
    a, b = 0.0, k + 10 * math.sqrt(k) + 10
    for _ in range(80):
        m = (a + b) / 2
        if cdf(k, m) > alpha / 2: a = m
        else: b = m
    return k / exposure, lo / exposure, ((a + b) / 2) / exposure


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def auc_ci(pos, neg, B=400, seed=0):
    rng = np.random.default_rng(seed); pos, neg = np.asarray(pos), np.asarray(neg)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return auc(pos, neg), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def roc_points(pos, neg, n=40):
    th = np.unique(np.r_[np.quantile(np.r_[pos, neg], np.linspace(0, 1, n)), np.inf])
    return [[float((neg >= t).mean()), float((pos >= t).mean())] for t in th]


def med_ci(x, B=400, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0: return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(x, len(x))) for _ in range(B)]
    return float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------- core computations
def calibrate(cal_runs, q=0.99):
    th = {}
    for i, n in enumerate(DET_ORDER):
        mx = np.array([r["S"][60:, i].max() for r in cal_runs])
        th[n] = float(np.quantile(mx, q, method="higher"))
        th[n] = max(th[n], 1e-6)
    return th


def fused(S, th):
    return (S / np.array([th[n] for n in DET_ORDER])).max(1)


def window(r, X=None):
    o = r["onset"]; return slice(o, o + WIN)


def episodes(a):
    a = np.asarray(a, bool); return int((a[1:] & ~a[:-1]).sum() + (1 if a[0] else 0))


def first_alert(a, onset):
    idx = np.flatnonzero(a[onset:onset + WIN]); return int(idx[0]) if len(idx) else None


def evaluate_set(clean_runs, attack_runs, th, B=400):
    out = dict(detect={}, delay={}, auc={}, roc={}, fa={}, loo={})
    names = DET_ORDER + ["fused"]
    th_arr = np.array([th[n] for n in DET_ORDER])

    def col(r, name):
        if name == "fused": return fused(r["S"], th)
        return r["S"][:, DET_ORDER.index(name)]

    def alarm(r, name):
        return latch_series(col(r, name), 1.0 if name == "fused" else th[name])

    # false alarms on clean runs
    hours = sum((len(r["S"]) - 60) for r in clean_runs) / 3600.0
    for n in names:
        k = sum(episodes(alarm(r, n)[60:]) for r in clean_runs)
        out["fa"][n] = dict(events=k, hours=hours, rate_per_h=poisson_ci(k, hours))
    cneg = {n: np.array([col(r, n)[window(r)].max() for r in clean_runs]) for n in names}
    for sc in ATTACKS:
        runs = [r for r in attack_runs if r["scenario"] == sc]
        for n in names:
            hit, delays = [], []
            for r in runs:
                a = alarm(r, n); d = first_alert(a, r["onset"]); hit.append(d is not None)
                if d is not None: delays.append(d)
            pos = np.array([col(r, n)[window(r)].max() for r in runs])
            out["detect"].setdefault(sc, {})[n] = wilson(sum(hit), len(hit))
            out["delay"].setdefault(sc, {})[n] = dict(zip(("median", "lo", "hi"), med_ci(delays, B))) | dict(n_detected=len(delays))
            out["auc"].setdefault(sc, {})[n] = auc_ci(pos, cneg[n], B)
            out["roc"].setdefault(sc, {})[n] = roc_points(pos, cneg[n])
        # leave-one-out fused detection
        for leave in DET_ORDER:
            keep = [i for i, d in enumerate(DET_ORDER) if d != leave]
            hit = [bool((r["S"][window(r)][:, keep] / th_arr[keep] > 1).any()) for r in runs]
            out["loo"].setdefault(sc, {})[leave] = wilson(sum(hit), len(hit))
    return out


def build_windows(runs, per_run, rng, attack_min=30):
    X, y = [], []
    for r in runs:
        ci = SCENARIOS.index(r["scenario"]); N = len(r["S"])
        lo = (r["onset"] + attack_min) if ci else 90
        for _ in range(per_run):
            t = int(rng.integers(lo, N)); sl = slice(t - WINDOW + 1, t + 1)
            X.append(window_features(r["S"][sl], r["R"][sl])); y.append(ci)
    return np.array(X), np.array(y)


def classifier_eval(model, test_runs):
    X, y = build_windows(test_runs, 1, np.random.default_rng(99))
    p = clf.predict(model, X); k = len(SCENARIOS)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, p): cm[a, b] += 1
    acc = wilson(int((p == y).sum()), len(y))
    rec = {SCENARIOS[i]: wilson(int(cm[i, i]), int(cm[i].sum())) for i in range(k)}
    return dict(accuracy=acc, confusion=cm.tolist(), recall=rec, n=len(y))


# ---------------------------------------------------------------- driver
def main(out_dir, n_cal=200, n_clean=200, n_att=100, n_train=60, n_shift=60, procs=2, quick=False, write_defaults=True):
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    def stamp(m): print("[%5.0fs] %s" % (time.time() - t0, m), flush=True)
    stamp("calibration runs"); cal = run_jobs([("clean", 1000 + i, "nominal") for i in range(n_cal)], procs)
    th = calibrate(cal); stamp("thresholds %s" % {k: round(v, 2) for k, v in th.items()})
    stamp("training runs"); tr = run_jobs([(sc, 2000 + 100 * j + i, "nominal") for j, sc in enumerate(SCENARIOS) for i in range(n_train)], procs)
    X, y = build_windows(tr, 6, np.random.default_rng(1)); model = clf.train(X, y, len(SCENARIOS))
    stamp("test runs")
    clean_t = run_jobs([("clean", 5000 + i, "nominal") for i in range(n_clean)], procs)
    att_t = run_jobs([(sc, 5000 + 1000 * (j + 1) + i, "nominal") for j, sc in enumerate(ATTACKS) for i in range(n_att)], procs)
    ev = evaluate_set(clean_t, att_t, th)
    ce = classifier_eval(model, clean_t[:n_att] + att_t)
    stamp("shifted-profile runs")
    clean_s = run_jobs([("clean", 7000 + i, "shifted") for i in range(n_shift)], procs)
    att_s = run_jobs([(sc, 7000 + 1000 * (j + 1) + i, "shifted") for j, sc in enumerate(ATTACKS) for i in range(n_shift)], procs)
    ev_s = evaluate_set(clean_s, att_s, th); ce_s = classifier_eval(model, clean_s + att_s)
    res = dict(thresholds=th, nominal=ev, shifted=ev_s, classifier=ce, classifier_shifted=ce_s, classes=SCENARIOS,
               config=dict(n_cal=n_cal, n_clean=n_clean, n_att=n_att, n_train=n_train, n_shift=n_shift, win=WIN, steps=N_STEPS,
                           calib_quantile=0.99, quick=quick, elapsed_s=round(time.time() - t0)))
    with open(os.path.join(out_dir, "results.json"), "w") as f: json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, "results.md"), "w") as f: f.write(report_md(res))
    if write_defaults:
        with open(os.path.join(HERE, "default_thresholds.json"), "w") as f:
            json.dump(dict(thresholds=th, note="calibrated on %d clean SYNTHETIC runs, 99th pct of per-run max score" % n_cal), f, indent=1)
        clf.save(model, SCENARIOS, os.path.join(HERE, "default_classifier.json"))
    stamp("done"); return res


# ---------------------------------------------------------------- report
def _c(t, d=2): return "%.*f [%.*f, %.*f]" % (d, t[0], d, t[1], d, t[2])


def report_md(res):
    cf = res["config"]; L = []
    L += ["# GNSS-Sentinel evaluation (SYNTHETIC DATA ONLY)", "",
          "Generated by `python -m gnss_sentinel evaluate`. All data come from the built-in synthetic generator; the detectors, "
          "thresholds and classifier were designed and tuned against that same generator family, so every number below is "
          "**optimistic** and says nothing about real receivers. 95% intervals: Wilson (proportions), exact Poisson (rates), "
          "percentile bootstrap over runs (AUC, medians).", "",
          "Config: %d calibration clean runs, %d test clean runs, %d test runs per attack, %d training runs per class, %d shifted-profile runs per class; "
          "720 epochs/run at 1 Hz, onset 180-300 s, evaluation window = 420 s after onset; thresholds = 99th percentile of per-run "
          "maximum score on calibration data. The first 60 s of each run are baseline learning." % (cf["n_cal"], cf["n_clean"], cf["n_att"], cf["n_train"], cf["n_shift"]), "",
          "Thresholds: " + ", ".join("%s=%.2f" % kv for kv in res["thresholds"].items()), ""]
    names = DET_ORDER + ["fused"]
    for tag, key in (("nominal generator (held-out seeds)", "nominal"), ("SHIFTED generator (noisier, heavier tails; nominal thresholds)", "shifted")):
        ev = res[key]
        L += ["## Results: " + tag, "", "### Detection probability within 420 s of onset (ablation: which detector catches which scenario)", "",
              "| scenario | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for sc in ATTACKS:
            L.append("| %s | " % sc + " | ".join("%.2f [%.2f,%.2f]" % tuple(ev["detect"][sc][n]) for n in names) + " |")
        L += ["", "### Median detection delay in seconds after onset [95% CI] (only runs that were detected; n detected in parentheses)", "",
              "| scenario | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for sc in ATTACKS:
            cells = []
            for n in names:
                d = ev["delay"][sc][n]
                cells.append("n/a (0)" if d["n_detected"] == 0 else "%.0f [%.0f,%.0f] (%d)" % (d["median"], d["lo"], d["hi"], d["n_detected"]))
            L.append("| %s | " % sc + " | ".join(cells) + " |")
        L += ["", "### Run-level ROC AUC [95% CI] (score = max over the 420 s window; negatives = clean runs, same window)", "",
              "| scenario | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for sc in ATTACKS:
            L.append("| %s | " % sc + " | ".join(_c(ev["auc"][sc][n]) for n in names) + " |")
        L += ["", "### False-alarm rate on clean runs (alert episodes per hour [95% exact Poisson CI])", "",
              "| detector | episodes | hours | rate/h |", "|---|---|---|---|"]
        for n in names:
            f = ev["fa"][n]; L.append("| %s | %d | %.1f | %s |" % (n, f["events"], f["hours"], _c(f["rate_per_h"], 3)))
        L += ["", "### Leave-one-out: fused detection probability with one detector removed (shows each detector's unique contribution)", "",
              "| scenario | all | " + " | ".join("-" + n for n in DET_ORDER) + " |", "|---|---|" + "---|" * len(DET_ORDER)]
        for sc in ATTACKS:
            L.append("| %s | %.2f | " % (sc, ev["detect"][sc]["fused"][0]) + " | ".join("%.2f" % ev["loo"][sc][n][0] for n in DET_ORDER) + " |")
        L.append("")
    for tag, key in (("nominal", "classifier"), ("shifted", "classifier_shifted")):
        c = res[key]
        L += ["## Softmax classifier, one 30 s window per run, %s generator" % tag, "",
              "Accuracy %s (n=%d). Rows = true class, columns = predicted: %s" % (_c(c["accuracy"]), c["n"], ", ".join(res["classes"])), "", "```"]
        for i, row in enumerate(c["confusion"]): L.append("%-14s %s" % (res["classes"][i], " ".join("%4d" % v for v in row)))
        L += ["```", "", "Per-class recall: " + "; ".join("%s %s" % (k, _c(v)) for k, v in c["recall"].items()), ""]
    L += ["## Honest reading", "",
          "* Synthetic, same-family data: thresholds and classifier were fitted on the same generator they are tested on (different seeds). Expect real-world performance to be materially worse, especially false-alarm rates.",
          "* The shifted-profile section shows the sensitivity to modest environment change alone; real receivers also differ in AGC/C/N0 reporting, tracking loops, antenna patterns, multipath and clock steering.",
          "* Scenario severities are drawn from ranges chosen by the author; weak-end cases (small clock steps, slow drag, weak CW) are not detectable by design and lower the detection rates.",
          "* Detection rate of 1.00 on a synthetic scenario means 'the signature I modelled is separable', not 'this attack is always detected'. A spoofer that keeps per-satellite power structure, is geometry-consistent and also shifts Doppler is outside the model.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
