#!/usr/bin/env python3
"""Generated P6 Look-Locker inputs for tools/regress_identity.sh (P7 plan, Task 0).

    regress_p6_inputs.py <p6_ll fixture dir> <output dir>

Writes two variants of p6_ll (two cycles of twelve readouts, PASL, separate M0):

- p6-ll-legacy.{json,tsv,toml}: one readout per cycle (rows 0 and 12) and the scalar FlipAngle,
  which takes the legacy dispatch (P5's single-readout gradient-echo path);
- p6-ll-m0.{json,tsv,toml}: an included m0scan row between the two cycles, every per-row array
  given its entry for it, and the overlay's [m0] table dropped (an included M0 takes none).

Every substitution is checked against the source, so a change that silently does nothing fails
here rather than giving the gate a second copy of p6_ll.
"""
import json
import re
import sys
from pathlib import Path

PER_ROW = ("PostLabelingDelay", "FlipAngle", "RepetitionTimePreparation", "VascularCrushingVENC")


def fail(msg):
    sys.exit(f"regress_p6_inputs: {msg}")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    side = json.loads((src / "asl.json").read_text())
    rows = (src / "aslcontext.tsv").read_text().split()
    overlay = (src / "overlay.toml").read_text()
    if rows[0] != "volume_type":
        fail("aslcontext.tsv has no volume_type header")
    rows = rows[1:]
    n = len(rows)
    if n != 24 or rows[:12] != ["control"] * 12 or rows[12:] != ["label"] * 12:
        fail(f"p6_ll is no longer two cycles of twelve readouts ({n} rows)")
    if not isinstance(side["FlipAngle"], (int, float)):
        fail("p6_ll's FlipAngle is no longer a scalar")
    arrays = [k for k in PER_ROW if isinstance(side.get(k), list)]
    if arrays != ["PostLabelingDelay"]:
        fail(f"p6_ll's per-row arrays changed: {arrays}")
    if side["M0Type"] != "Separate" or "[m0]" not in overlay:
        fail("p6_ll no longer has a separate M0 with an [m0] table")
    if not re.search(r"(?m)^readouts_per_cycle = 12$", overlay):
        fail("p6_ll's overlay no longer sets readouts_per_cycle = 12")
    pld = side["PostLabelingDelay"]

    # one readout per cycle
    leg = dict(side, PostLabelingDelay=[pld[0], pld[12]])
    leg_ov = overlay.replace("readouts_per_cycle = 12", "readouts_per_cycle = 1")
    leg_rows = [rows[0], rows[12]]
    if leg_ov == overlay or len(leg["PostLabelingDelay"]) != 2 or leg_rows != ["control", "label"]:
        fail("the legacy variant did not apply")

    # an included m0scan between the cycles
    m0 = dict(side, M0Type="Included", PostLabelingDelay=pld[:12] + [0.0] + pld[12:])
    m0_rows = rows[:12] + ["m0scan"] + rows[12:]
    m0_ov = re.sub(r"(?ms)^\[m0\]\n.*?(?=^\[|\Z)", "", overlay)
    if "[m0]" in m0_ov or len(m0["PostLabelingDelay"]) != 25 or m0_rows.count("m0scan") != 1:
        fail("the included-M0 variant did not apply")
    if "[look_locker]" not in m0_ov or "readouts_per_cycle = 12" not in m0_ov:
        fail("dropping [m0] removed more than the [m0] table")

    out.mkdir(parents=True, exist_ok=True)
    for name, s, r, o in (("p6-ll-legacy", leg, leg_rows, leg_ov), ("p6-ll-m0", m0, m0_rows, m0_ov)):
        (out / f"{name}.json").write_text(json.dumps(s, indent=1) + "\n")
        (out / f"{name}.tsv").write_text("volume_type\n" + "".join(f"{x}\n" for x in r))
        (out / f"{name}.toml").write_text(o)


if __name__ == "__main__":
    main()
