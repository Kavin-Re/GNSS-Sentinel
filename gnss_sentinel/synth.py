"""Synthetic receiver-measurement generator -- TEST DATA ONLY.

Purpose: produce labelled *abstract feature records* (C/N0, AGC, residuals, ...) so that the passive detectors in
detectors.py can be unit-tested and evaluated. It does NOT generate RF/IQ samples, has no frequency or waveform
model, and nothing here can be transmitted. Attack scenarios are statistical descriptions of what a receiver
*would report* under each condition, written from public descriptions of jamming/spoofing effects. Real receivers
differ (see README "Limits").

Model summary (1 Hz epochs)
  * ~15 satellites with slowly moving elevation; tracked set = highest 14 above 7 deg with C/N0 hysteresis (22/26).
  * Clean C/N0 = 35 + 13 sin(el)^0.7 + per-satellite offset + AR(1) + white noise, quantised to 1 dB-Hz.
  * Pseudorange error eps_i = noise_i/sin(el) + multipath_i + attack offset_i. Position, clock bias and post-fit
    residuals come from one weighted least-squares fit of eps, so they are mutually consistent.
  * Noise realisation depends only on `seed`, not on the scenario (paired clean/attack runs), attack parameters
    come from a separate RNG stream.
Scenarios: clean, jam_ramp, cw, meacon, spoof_uniform, drag (+ sat_fault, used only in unit tests)."""
import numpy as np
from .epoch import Epoch

C_M_PER_US = 299.792458
SCENARIOS = ["clean", "jam_ramp", "cw", "meacon", "spoof_uniform", "drag"]
ATTACKS = SCENARIOS[1:]
NCAND, NCH = 18, 14
PROFILES = {
    "nominal": dict(cn0_static=1.0, cn0_ar=0.6, cn0_white=0.7, mp=1.0, sig0=1.2, tail_df=None, vel=0.05,
                    agc=0.25, noise=0.2, clk_rw=0.01),
    # deliberately different environment (stress / generator-shift test): noisier, heavier tails, more multipath
    "shifted": dict(cn0_static=2.2, cn0_ar=1.2, cn0_white=1.4, mp=2.5, sig0=2.0, tail_df=4, vel=0.12,
                    agc=0.5, noise=0.4, clk_rw=0.03),
}


def _unit(v):
    return v / np.linalg.norm(v)


def generate(scenario="clean", seed=0, n=720, onset=None, profile="nominal", sat_fault_m=0.0, **overrides):
    """Return (list[Epoch], meta). `onset` (epoch index) defaults to a seed-dependent value in [180, 300]."""
    if scenario not in SCENARIOS + ["sat_fault"]:
        raise ValueError("unknown scenario %r" % scenario)
    P = {**PROFILES[profile], **overrides}
    rng = np.random.default_rng(seed)                 # environment + noise (scenario independent)
    ra = np.random.default_rng(seed + 10_000_019)     # attack parameters
    if onset is None:
        onset = int(ra.integers(180, 301))
    meta = dict(scenario=scenario, seed=seed, onset=onset, profile=profile)

    # ---- attack parameters (always drawn so the stream is identical regardless of branch) ----
    Jmax, ramp = ra.uniform(15, 38), ra.uniform(60, 200)
    Jcw = ra.uniform(12, 30); hit = ra.random(NCAND) < 0.4
    hitf, lowf = ra.uniform(0.4, 1.0, NCAND), ra.uniform(0, 0.15, NCAND)
    cw_per, cw_ph = ra.uniform(120, 300), ra.uniform(0, 6.28)
    tau_us = 10 ** ra.uniform(np.log10(0.03), np.log10(2.0)); boost_m = ra.uniform(2, 7)
    xs = ra.normal(0, ra.uniform(1, 8), NCAND)
    P_sp, agc_sp = ra.uniform(42, 50), ra.uniform(0.5, 2.5)
    off_mag = ra.lognormal(np.log(5), 0.8) * (ra.random() < 0.8)
    off_dir = _unit(ra.normal(0, 1, 3) * [1, 1, 0.3]); clk_off = ra.normal(0, 3)
    rate = float(np.exp(ra.uniform(np.log(0.15), np.log(1.0)))); drag_dir = _unit(ra.normal(0, 1, 3) * [1, 1, 0.2])
    drag_boost = ra.uniform(0, 2)
    if scenario == "jam_ramp": meta.update(Jmax_dB=Jmax, ramp_s=ramp)
    if scenario == "cw": meta.update(J_cw_dB=Jcw)
    if scenario == "meacon": meta.update(delay_us=tau_us, clock_jump_m=C_M_PER_US * tau_us, boost_db=boost_m)
    if scenario == "spoof_uniform": meta.update(power_dbhz=P_sp, takeover_offset_m=off_mag)
    if scenario == "drag": meta.update(drag_rate_mps=rate)

    # ---- environment state ----
    sinel0 = rng.uniform(-0.2, 1.0, NCAND)
    el = np.degrees(np.arcsin(sinel0)); az = rng.uniform(0, 360, NCAND)
    el_rate = rng.uniform(-0.012, 0.012, NCAND); az_rate = rng.uniform(-0.01, 0.01, NCAND)
    prn_all = rng.permutation(32)[:NCAND] + 1
    cn_static = rng.normal(0, P["cn0_static"], NCAND)
    ar_c = np.zeros(NCAND); ar_m = np.zeros(NCAND); agc_ar = 0.0; noi_ar = 0.0
    tracking = el >= 7
    sp0, hd = rng.uniform(3, 15), rng.uniform(0, 2 * np.pi)
    vel = np.array([sp0 * np.cos(hd), sp0 * np.sin(hd), 0.0]); acc = np.zeros(3); pos = np.zeros(3)
    clk_b, clk_d = 0.0, rng.normal(0, 0.3)
    fault_idx = None
    out = []
    for k in range(n):
        t = float(k)
        # ---- always-drawn randomness (fixed order => paired runs) ----
        z_ar_c = rng.normal(0, 1, NCAND); z_white = rng.normal(0, 1, NCAND); z_ar_m = rng.normal(0, 1, NCAND)
        z_rng = rng.normal(0, 1, NCAND); z_acc = rng.normal(0, 1, 3); z_vel = rng.normal(0, 1, 3)
        z_agc, z_noi, z_clk, z_agcw, z_noiw = rng.normal(0, 1, 5)
        z_cnsp, z_mpsp, z_mpn = rng.normal(0, 1, NCAND), rng.normal(0, 1, NCAND), rng.normal(0, 1, NCAND)
        z_jit = rng.normal(0, 1, NCAND)
        if P["tail_df"]:
            df = P["tail_df"]; tl = rng.standard_t(df, NCAND) / np.sqrt(df / (df - 2))
        else:
            tl = z_rng
        # ---- dynamics ----
        el = el + el_rate; az = (az + az_rate) % 360
        acc = np.clip(0.9 * acc + z_acc * np.array([0.35, 0.35, 0.1]), -3, 3)
        vnew = vel + acc; vnew[2] *= 0.9
        pos = pos + 0.5 * (vel + vnew); vel = vnew
        clk_d += P["clk_rw"] * z_clk; clk_b += clk_d
        sinel = np.sin(np.radians(el)).clip(0.05, 1)
        ar_c = 0.97 * ar_c + np.sqrt(1 - 0.97 ** 2) * P["cn0_ar"] * z_ar_c
        cn_clean = 35 + 13 * sinel ** 0.7 + cn_static + ar_c + P["cn0_white"] * z_white
        ar_m = 0.95 * ar_m + np.sqrt(1 - 0.95 ** 2) * z_ar_m
        mp_true = ar_m * P["mp"] * (0.2 + 1.2 * np.exp(-el / 15))
        agc_ar = 0.98 * agc_ar + 0.03 * z_agcw; noi_ar = 0.98 * noi_ar + 0.03 * z_noiw
        agc = agc_ar + P["agc"] * z_agc; noi = noi_ar + P["noise"] * z_noi
        # ---- attack effects ----
        on = k >= onset
        cn = cn_clean.copy()
        drop = np.zeros(NCAND)
        fade = float(np.clip((k - onset) / 5.0, 0, 1)) if on else 0.0
        dpos = np.zeros(3); dclk = 0.0; extra = np.zeros(NCAND); mp_rep = None
        if scenario == "jam_ramp" and on:
            J = -15 + (Jmax + 15) * min(1.0, (k - onset) / ramp)
            rise = 10 * np.log10(1 + 10 ** (J / 10))
            drop = rise + 0.3 * z_jit; agc += min(rise, 28) * 0.9; noi += rise
        elif scenario == "cw" and on:
            mod = 1 + 0.15 * np.sin(2 * np.pi * (k - onset) / cw_per + cw_ph)
            frac = np.where(hit, hitf, lowf)
            drop = np.minimum(fade * frac * Jcw * 0.8 * mod, 22) + 0.2 * z_jit
            agc += fade * 0.15 * Jcw * mod; noi += fade * 0.5 * 0.15 * Jcw * mod
        elif scenario == "meacon" and on:
            cn = cn + fade * boost_m; agc += fade * 0.25 * boost_m
            dclk = fade * C_M_PER_US * tau_us; extra = fade * xs
        elif scenario == "spoof_uniform" and on:
            cn = (1 - fade) * cn + fade * (P_sp + 0.4 * z_cnsp); agc += fade * agc_sp
            dpos = fade * off_mag * off_dir; dclk = fade * clk_off
            mp_rep = fade
        elif scenario == "drag" and on:
            cn = cn + fade * drag_boost
            dpos = rate * (k - onset) * drag_dir
        cn_eff = np.round(cn - drop)
        tracking = np.where(tracking, cn_eff > 22, cn_eff > 26) & (el >= 7)
        idx = np.flatnonzero(tracking); idx = idx[np.argsort(-el[idx])][:NCH]
        m = len(idx)
        e = Epoch(t=t, n_sat=m, label=scenario if (on and scenario != "clean") else "clean")
        e.prn = prn_all[idx].astype(int); e.el = el[idx].copy(); e.az = az[idx].copy(); e.cn0 = cn_eff[idx]
        e.agc_db = round(float(agc) * 4) / 4; e.noise_db = round(float(noi) * 4) / 4
        if scenario == "sat_fault" and on and m:
            if fault_idx is None: fault_idx = int(e.prn[0]); meta["fault_prn"] = fault_idx
        # ---- least-squares solution ----
        dm = np.clip(cn_clean - cn_eff, 0, 30)
        sig = P["sig0"] / np.maximum(sinel, 0.35) * 10 ** (dm / 20)
        mpv = mp_true.copy()
        if mp_rep is not None:
            mpv = (1 - mp_rep) * mp_true + mp_rep * 0.1 * z_mpsp
        if m >= 4:
            los = np.stack([np.sin(np.radians(az[idx])) * np.cos(np.radians(el[idx])),
                            np.cos(np.radians(az[idx])) * np.cos(np.radians(el[idx])),
                            np.sin(np.radians(el[idx]))], 1)
            G = np.c_[-los, np.ones(m)]
            eps = tl[idx] * sig[idx] + mpv[idx] + extra[idx] + G @ np.r_[dpos, dclk]
            if scenario == "sat_fault" and on and fault_idx is not None:
                eps = eps + sat_fault_m * (e.prn == fault_idx)
            w = 1 / sig[idx] ** 2
            A = G.T @ (G * w[:, None])
            if np.linalg.cond(A) < 1e7:
                x = np.linalg.solve(A, G.T @ (w * eps))
                e.res = eps - G @ x
                e.pos = pos + x[:3]; e.clock_m = clk_b + x[3]
                e.vel = vel + P["vel"] * 10 ** (min(dm[idx].mean(), 20) / 20) * z_vel
            else:
                e.res = np.full(m, np.nan)
        else:
            e.res = np.full(m, np.nan)
        e.mp = mpv[idx] + 0.05 * z_mpn[idx]
        out.append(e)
    return out, meta
