#!/usr/bin/env python3
"""Write a timed rigid-motion trace (P10): the motion TSV with a `time` column.

    python tools/motion_trace.py --rate 100 --duration 190 --out nod.tsv \
        --nod start=58.8,rise=0.4,hold=4.2,fall=0.4,trans=0:2:3,rot=0.06:0:0 \
        --drift trans=0:0:0.5

    python tools/motion_trace.py --from physio_motion.tsv --out timed.tsv

The columns are `time` (s), `trans_x/y/z` (mm) and `rot_x/y/z` (radians), the format aslscan's
`[motion] trajectory` and TRXScan's `--motion` read, rotations about the field-of-view centre.
Samples are at `start + i / rate` for `i = 0 .. round(duration * rate)`, both ends included.

Components add:

- `--nod start=,rise=,hold=,fall=,trans=x:y:z,rot=x:y:z[,shape=cosine|linear]`: a rise from zero to
  the amplitude, a hold, and a fall back. Edges are half cosines by default; `linear` makes the
  pose between two samples equal what one jump per shot accumulates, for checks against an events
  file.
- `--step start=,rise=,trans=,rot=[,shape=]`: a rise with no fall.
- `--drift trans=,rot=`: a linear ramp from zero at the first sample to the amplitude at the last.

`--from` rewrites a trace timed by a JSON sidecar (`SamplingFrequency` in Hz, `StartTime` in s,
default 0: the BIDS physio convention) into the `time` column form. Standard library only.
"""
import argparse
import csv
import json
import math
import os
import sys

COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def parse_vec(s, what):
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"{what} = {s!r}: expected x:y:z")
    v = [float(p) for p in parts]
    if not all(math.isfinite(x) for x in v):
        raise ValueError(f"{what} = {s!r}: not finite")
    return v


def parse_component(kind, spec):
    """`key=value,...` into a dict with floats, vectors and the shape."""
    fields = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"--{kind} {spec!r}: {item!r} is not key=value")
        k, v = item.split("=", 1)
        fields[k.strip()] = v.strip()
    allowed = {"nod": {"start", "rise", "hold", "fall", "trans", "rot", "shape"},
               "step": {"start", "rise", "trans", "rot", "shape"},
               "drift": {"trans", "rot"}}[kind]
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"--{kind}: unknown keys {sorted(unknown)}; expected {sorted(allowed)}")
    c = {"kind": kind,
         "trans": parse_vec(fields.get("trans", "0:0:0"), f"--{kind} trans"),
         "rot": parse_vec(fields.get("rot", "0:0:0"), f"--{kind} rot"),
         "shape": fields.get("shape", "cosine")}
    if c["shape"] not in ("cosine", "linear"):
        raise ValueError(f"--{kind} shape = {c['shape']!r}: expected cosine or linear")
    required = {"nod": ("start", "rise", "fall"), "step": ("start", "rise"), "drift": ()}[kind]
    for k in required:
        if k not in fields:
            raise ValueError(f"--{kind} needs {k}=")
    for k in ("start", "rise", "hold", "fall"):
        if k not in allowed:
            continue
        x = float(fields.get(k, "0"))  # hold defaults to 0
        if not math.isfinite(x) or (k != "start" and x < 0):
            raise ValueError(f"--{kind} {k} = {x}: not a finite{'' if k == 'start' else ' non-negative'} number")
        c[k] = x
    return c


def edge(u, shape):
    """0 at u <= 0, 1 at u >= 1, half cosine or linear between."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u if shape == "linear" else 0.5 * (1.0 - math.cos(math.pi * u))


def profile(c, t, t0, t1):
    """The component's scale (0..1) at time t; t0, t1 the trace's span (for the drift)."""
    if c["kind"] == "drift":
        return 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    s = c["start"]
    if t < s:
        return 0.0
    up = 1.0 if c["rise"] == 0.0 else edge((t - s) / c["rise"], c["shape"])
    if c["kind"] == "step":
        return up
    down_at = s + c["rise"] + c["hold"]
    if t < down_at:
        return up
    return 1.0 - (1.0 if c["fall"] == 0.0 else edge((t - down_at) / c["fall"], c["shape"]))


def span(c, t0, t1):
    if c["kind"] == "drift":
        return t0, t1
    end = c["start"] + c["rise"] + (c["hold"] + c["fall"] if c["kind"] == "nod" else 0.0)
    return c["start"], end


def synthesize(rate, duration, start, components):
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"--rate {rate}: not a positive number")
    if not (math.isfinite(duration) and duration > 0):
        raise ValueError(f"--duration {duration}: not a positive number")
    n = int(round(duration * rate)) + 1
    times = [start + i / rate for i in range(n)]
    t0, t1 = times[0], times[-1]
    rows = []
    for t in times:
        pose = [0.0] * 6
        for c in components:
            f = profile(c, t, t0, t1)
            amp = c["trans"] + c["rot"]
            for k in range(6):
                pose[k] += f * amp[k]
        rows.append([t] + pose)
    return rows


def from_sidecar(path):
    """Rows `[time] + pose` of a sidecar-timed trace."""
    side = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(side):
        raise ValueError(f"{path}: no sidecar {side}")
    with open(side) as f:
        meta = json.load(f)
    if "SamplingFrequency" not in meta:
        raise ValueError(f"{side}: no SamplingFrequency")
    hz = float(meta["SamplingFrequency"])
    t0 = float(meta.get("StartTime", 0.0))
    if not (math.isfinite(hz) and hz > 0):
        raise ValueError(f"{side}: SamplingFrequency {hz} is not positive")
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = [h.strip() for h in next(r)]
        if "time" in header:
            raise ValueError(f"{path}: has a time column and a sidecar SamplingFrequency; two clocks")
        try:
            idx = [header.index(c) for c in COLS]
        except ValueError as e:
            raise ValueError(f"{path}: missing column ({e})") from None
        rows = []
        for i, line in enumerate(x for x in r if any(cell.strip() for cell in x)):
            vals = []
            for c in idx:
                cell = line[c].strip() if c < len(line) else ""
                v = 0.0 if cell == "n/a" else float(cell)
                if not math.isfinite(v):
                    raise ValueError(f"{path} row {i}: {cell!r} is not finite")
                vals.append(v)
            rows.append([t0 + i / hz] + vals)
    return rows


def write(rows, out):
    with open(out, "w", newline="") as f:
        f.write("\t".join(["time"] + COLS) + "\n")
        for r in rows:
            f.write("\t".join(repr(float(x)) for x in r) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, help="samples per second")
    ap.add_argument("--duration", type=float, help="seconds")
    ap.add_argument("--start", type=float, default=0.0, help="time of the first sample (s)")
    ap.add_argument("--nod", action="append", default=[])
    ap.add_argument("--step", action="append", default=[])
    ap.add_argument("--drift", action="append", default=[])
    ap.add_argument("--from", dest="source", help="a sidecar-timed trace to rewrite with a time column")
    a = ap.parse_args(argv)
    if a.source:
        if a.nod or a.step or a.drift or a.rate or a.duration:
            ap.error("--from takes no components, --rate or --duration")
        rows = from_sidecar(a.source)
    else:
        if a.rate is None or a.duration is None:
            ap.error("--rate and --duration are required (or --from)")
        comps = ([parse_component("nod", s) for s in a.nod] + [parse_component("step", s) for s in a.step]
                 + [parse_component("drift", s) for s in a.drift])
        rows = synthesize(a.rate, a.duration, a.start, comps)
        for c in comps:
            s0, s1 = span(c, rows[0][0], rows[-1][0])
            print(f"{c['kind']}: {s0:.6g} to {s1:.6g} s, trans {c['trans']} mm, rot {c['rot']} rad")
    write(rows, a.out)
    print(f"trace: {len(rows)} samples, {rows[0][0]:.6g} to {rows[-1][0]:.6g} s -> {a.out}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"motion_trace.py: {e}")
