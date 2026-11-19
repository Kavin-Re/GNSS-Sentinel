"""Log formats.

1) CSV (native, documented in README): one row per epoch, header-driven, '#' comment lines allowed, empty cell = missing.
   Columns: t,n_sat,agc_db,noise_db,clock_m,e_m,n_m,u_m,ve_mps,vn_mps,vu_mps,label, then per satellite slot k=0..K-1:
   prn{k},el{k},az{k},cn0{k},res{k},mp{k}.  'label' is optional ground truth (synthetic logs only) and is never used by detectors.
2) NMEA 0183 sentences ($..GGA, $..RMC, $..GSV). Only the fields below are read; checksums are verified when present.
   GGA : time, lat, N/S, lon, E/W, fix quality, number of satellites used, ..., altitude
   RMC : time, status, lat, N/S, lon, E/W, speed over ground (knots), course over ground (deg true)
   GSV : total messages, message number, SVs in view, then groups of (PRN, elevation, azimuth, SNR[dB-Hz]); an optional
         trailing signal-ID field (NMEA 4.10+) is accepted.
   NMEA carries no AGC, clock bias, residuals or multipath, so those detectors report n/a on NMEA input. Any talker prefix
   (GP, GL, GA, GB, GN, GI, ...) is accepted by matching the 3-letter sentence type only; NavIC-specific behaviour is not modelled.
   Epoch grouping: an epoch starts at each new GGA/RMC UTC time; GSV sentences are attached to the epoch most recently opened
   (so receivers must emit GGA/RMC before that epoch's GSV, which is the usual order)."""
import csv, io, math
import numpy as np
from .epoch import Epoch, NAN

K_MAX = 16
BASE = ["t", "n_sat", "agc_db", "noise_db", "clock_m", "e_m", "n_m", "u_m", "ve_mps", "vn_mps", "vu_mps", "label"]
SLOT = ["prn", "el", "az", "cn0", "res", "mp"]
R_EARTH = 6378137.0
KN = 0.514444


def csv_header(K=K_MAX):
    return BASE + ["%s%d" % (s, k) for k in range(K) for s in SLOT]


def _fmt(x, p=3):
    return "" if (x is None or (isinstance(x, float) and not math.isfinite(x))) else ("%.*f" % (p, x))


def write_csv(f, epochs, K=K_MAX, header_comment=None):
    if header_comment:
        for line in header_comment.splitlines(): f.write("# " + line + "\n")
    w = csv.writer(f, lineterminator="\n"); w.writerow(csv_header(K))
    for e in epochs:
        row = [_fmt(e.t, 2), e.n_sat, _fmt(e.agc_db, 2), _fmt(e.noise_db, 2), _fmt(e.clock_m, 3)]
        row += [_fmt(float(x), 3) for x in e.pos] + [_fmt(float(x), 3) for x in e.vel] + [e.label]
        for k in range(K):
            if k < len(e.prn):
                row += [int(e.prn[k]), _fmt(float(e.el[k]), 1), _fmt(float(e.az[k]), 1), _fmt(float(e.cn0[k]), 1),
                        _fmt(float(e.res[k]), 3), _fmt(float(e.mp[k]), 3)]
            else:
                row += [""] * 6
        w.writerow(row)


def _num(s):
    s = s.strip() if s else ""
    if not s: return NAN
    try: return float(s)
    except ValueError: return NAN


def read_csv(lines):
    """Generator of Epoch from an iterable of text lines (file, stdin, list). Streams: yields as each row arrives."""
    cols = None; K = 0
    for line in lines:
        if not line.strip() or line.startswith("#"): continue
        row = next(csv.reader([line]))
        if cols is None:
            cols = {c.strip(): i for i, c in enumerate(row)}
            if "t" not in cols: raise ValueError("CSV header must contain column 't'")
            K = 0
            while "prn%d" % K in cols: K += 1
            continue
        g = lambda name: _num(row[cols[name]]) if name in cols and cols[name] < len(row) else NAN
        e = Epoch(t=g("t"))
        e.agc_db, e.noise_db, e.clock_m = g("agc_db"), g("noise_db"), g("clock_m")
        e.pos = np.array([g("e_m"), g("n_m"), g("u_m")]); e.vel = np.array([g("ve_mps"), g("vn_mps"), g("vu_mps")])
        if "label" in cols and cols["label"] < len(row): e.label = row[cols["label"]].strip()
        P, EL, AZ, CN, RS, MP = [], [], [], [], [], []
        for k in range(K):
            p = g("prn%d" % k)
            if not math.isfinite(p): continue
            P.append(int(p)); EL.append(g("el%d" % k)); AZ.append(g("az%d" % k)); CN.append(g("cn0%d" % k))
            RS.append(g("res%d" % k)); MP.append(g("mp%d" % k))
        e.prn = np.array(P, int); e.el, e.az, e.cn0 = np.array(EL), np.array(AZ), np.array(CN)
        e.res, e.mp = np.array(RS), np.array(MP)
        ns = g("n_sat"); e.n_sat = int(ns) if math.isfinite(ns) else int(np.isfinite(e.cn0).sum())
        if not math.isfinite(e.t): continue
        yield e


# ---------------------------------------------------------------- NMEA
def checksum(body):
    c = 0
    for ch in body: c ^= ord(ch)
    return "%02X" % c


def split_sentence(line):
    """Return list of fields (talker+type first) or None if malformed / checksum mismatch."""
    line = line.strip()
    if not line.startswith("$"): return None
    body, star, cs = line[1:].partition("*")
    if star:
        cs = cs.strip()[:2].upper()
        if cs != checksum(body): return None
    return body.split(",")


def _ang(s, hemi, deg_digits):
    if not s: return NAN
    d = float(s[:deg_digits]); m = float(s[deg_digits:])
    v = d + m / 60.0
    return -v if hemi in ("S", "W") else v


def _hms(s):
    h, m, sec = int(s[0:2]), int(s[2:4]), float(s[4:])
    return h * 3600 + m * 60 + sec


class NmeaReader:
    """Incremental NMEA -> Epoch. feed(line) returns a list of completed epochs; flush() at end of stream."""
    def __init__(self):
        self.cur = None; self.key = None; self.origin = None; self.day = 0; self.last_sod = None
        self.talkers = {}; self.bad = 0; self.nlines = 0

    def _new(self, tstr):
        sod = _hms(tstr)
        if self.last_sod is not None and sod < self.last_sod - 43200: self.day += 1
        self.last_sod = sod
        self.cur = dict(t=sod + 86400 * self.day, sats={}, n_sat=None, pos=None, vel=[NAN] * 3)
        self.key = tstr

    def _close(self):
        c, self.cur = self.cur, None
        if c is None: return []
        e = Epoch(t=c["t"])
        keys = sorted(c["sats"])
        e.prn = np.array([k[0] for k in keys], int)
        e.el = np.array([c["sats"][k][0] for k in keys]); e.az = np.array([c["sats"][k][1] for k in keys])
        e.cn0 = np.array([c["sats"][k][2] for k in keys])
        e.res = np.full(len(keys), NAN); e.mp = np.full(len(keys), NAN)
        tracked = int(np.isfinite(e.cn0).sum())
        e.n_sat = c["n_sat"] if c["n_sat"] is not None else tracked
        if c["pos"] is not None:
            lat, lon, alt = c["pos"]
            if self.origin is None: self.origin = (lat, lon, alt if math.isfinite(alt) else 0.0)
            la0, lo0, al0 = self.origin
            e.pos = np.array([math.radians(lon - lo0) * R_EARTH * math.cos(math.radians(la0)),
                              math.radians(lat - la0) * R_EARTH, (alt - al0) if math.isfinite(alt) else NAN])
        e.vel = np.array(c["vel"])
        return [e]

    def _tid(self, talker, prn):
        if talker not in self.talkers: self.talkers[talker] = len(self.talkers)
        return self.talkers[talker] * 1000 + prn

    def feed(self, line):
        self.nlines += 1
        f = split_sentence(line)
        if f is None:
            if line.strip().startswith("$"): self.bad += 1
            return []
        head = f[0]
        if len(head) < 5 or head.startswith("P"): return []
        typ, talker = head[-3:], head[:-3]
        out = []
        try:
            if typ in ("GGA", "RMC") and f[1]:
                if self.key != f[1] or self.cur is None:
                    out = self._close(); self._new(f[1])
                if typ == "GGA":
                    q = int(f[6] or 0)
                    if self.cur["n_sat"] is None and f[7]: self.cur["n_sat"] = int(f[7])
                    if q > 0 and f[2] and f[4]:
                        self.cur["pos"] = (_ang(f[2], f[3], 2), _ang(f[4], f[5], 3), float(f[9]) if f[9] else NAN)
                else:
                    if f[2] == "A" and f[7] and f[8]:
                        v = float(f[7]) * KN; c = math.radians(float(f[8]))
                        self.cur["vel"] = [v * math.sin(c), v * math.cos(c), NAN]
            elif typ == "GSV" and self.cur is not None:
                rest = f[4:]
                if len(rest) % 4 == 1: rest = rest[:-1]      # trailing signal ID (NMEA 4.10+)
                for i in range(0, len(rest) - 3, 4):
                    if not rest[i]: continue
                    prn = self._tid(talker, int(rest[i]))
                    el = float(rest[i + 1]) if rest[i + 1] else NAN; az = float(rest[i + 2]) if rest[i + 2] else NAN
                    snr = float(rest[i + 3]) if rest[i + 3] else NAN
                    old = self.cur["sats"].get((prn,))
                    if old is None or (math.isfinite(snr) and not (old[2] >= snr)):
                        self.cur["sats"][(prn,)] = (el, az, snr)
        except (ValueError, IndexError):
            self.bad += 1
        return out

    def flush(self):
        return self._close()


def read_nmea(lines):
    r = NmeaReader()
    for line in lines:
        for e in r.feed(line): yield e
    for e in r.flush(): yield e


def detect_format(first_lines):
    for l in first_lines:
        l = l.strip()
        if not l or l.startswith("#"): continue
        return "nmea" if l.startswith("$") else "csv"
    return "csv"


def read_log(lines, fmt="auto"):
    lines = iter(lines)
    if fmt == "auto":
        buf = []
        for l in lines:
            buf.append(l)
            if l.strip() and not l.startswith("#"): break
        fmt = detect_format(buf)
        import itertools
        lines = itertools.chain(buf, lines)
    return read_nmea(lines) if fmt == "nmea" else read_csv(lines)


# ---- NMEA writer (for demos/tests: converts a synthetic log to NMEA-like text) ----
def _sent(body): return "$%s*%s" % (body, checksum(body))


def write_nmea(f, epochs, lat0=20.0, lon0=78.0, alt0=300.0, talker="GP", t0=43200.0):
    for e in epochs:
        sod = (t0 + e.t) % 86400; hh, mm, ss = int(sod // 3600), int(sod % 3600 // 60), sod % 60
        ts = "%02d%02d%05.2f" % (hh, mm, ss)
        if e.has_fix:
            lat = lat0 + math.degrees(e.pos[1] / R_EARTH); lon = lon0 + math.degrees(e.pos[0] / (R_EARTH * math.cos(math.radians(lat0))))
            la = "%02d%08.5f,%s" % (int(abs(lat)), (abs(lat) - int(abs(lat))) * 60, "N" if lat >= 0 else "S")
            lo = "%03d%08.5f,%s" % (int(abs(lon)), (abs(lon) - int(abs(lon))) * 60, "E" if lon >= 0 else "W")
            sp = math.hypot(e.vel[0], e.vel[1]) / KN if np.isfinite(e.vel[:2]).all() else 0.0
            co = math.degrees(math.atan2(e.vel[0], e.vel[1])) % 360 if np.isfinite(e.vel[:2]).all() else 0.0
            f.write(_sent("%sRMC,%s,A,%s,%s,%.2f,%.1f,010100,,,A" % (talker, ts, la, lo, sp, co)) + "\n")
            f.write(_sent("%sGGA,%s,%s,%s,1,%02d,1.0,%.2f,M,0.0,M,," % (talker, ts, la, lo, e.n_sat, alt0 + e.pos[2])) + "\n")
        else:
            f.write(_sent("%sGGA,%s,,,,,0,%02d,,,M,,M,," % (talker, ts, e.n_sat)) + "\n")
        n = len(e.prn); msgs = max(1, (n + 3) // 4)
        for mi in range(msgs):
            parts = ["%sGSV" % talker, str(msgs), str(mi + 1), str(n)]
            for k in range(mi * 4, min(n, mi * 4 + 4)):
                snr = "%d" % round(e.cn0[k]) if np.isfinite(e.cn0[k]) else ""
                parts += ["%02d" % (e.prn[k] % 1000), "%02d" % round(e.el[k]), "%03d" % (round(e.az[k]) % 360), snr]
            f.write(_sent(",".join(parts)) + "\n")
