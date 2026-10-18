#!/usr/bin/env python3
"""Tests of tools/motion_trace.py (P10 plan, Task 2). Standard library only:

    python3 tools/test_motion_trace.py
"""
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_trace as mt  # noqa: E402


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def dump_text(text, path):
    with open(path, "w") as f:
        f.write(text)


def read(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines[0].split("\t"), [[float(x) for x in l.split("\t")] for l in lines[1:]]


class Synthesize(unittest.TestCase):
    def nod(self, spec):
        return mt.parse_component("nod", spec)

    def test_hold_equals_the_amplitude(self):
        c = self.nod("start=1,rise=0.4,hold=1,fall=0.4,trans=0:2:3,rot=0.06:0:0")
        rows = mt.synthesize(100, 4, 0.0, [c])
        hold = [r for r in rows if 1.4 <= r[0] <= 2.4]
        self.assertTrue(hold)
        for r in hold:
            self.assertEqual(r[1:], [0.0, 2.0, 3.0, 0.06, 0.0, 0.0])
        self.assertEqual(rows[0][1:], [0.0] * 6)
        self.assertEqual(rows[-1][1:], [0.0] * 6)

    def test_rise_is_monotonic_and_fall_returns(self):
        for shape in ("cosine", "linear"):
            c = self.nod(f"start=1,rise=0.5,hold=0.2,fall=0.5,trans=0:0:3,shape={shape}")
            rows = mt.synthesize(1000, 3, 0.0, [c])
            z = [r[3] for r in rows]
            rise = [z[i] for i, r in enumerate(rows) if 1.0 <= r[0] <= 1.5]
            self.assertTrue(all(b >= a for a, b in zip(rise, rise[1:])), shape)
            self.assertAlmostEqual(max(z), 3.0, places=12)
            self.assertEqual(z[-1], 0.0)

    def test_linear_nod_matches_cumulative_jumps(self):
        # P10 criterion 2: at the shot times, a linear rise equals k / 9 of the amplitude
        c = self.nod("start=58.8,rise=0.36,hold=4.24,fall=0.36,trans=0:2:3,rot=0.06:0:0,shape=linear")
        rows = mt.synthesize(1000, 70, 0.0, [c])
        by_t = {round(r[0], 9): r for r in rows}
        for k in range(10):
            r = by_t[round(58.8 + 0.04 * k, 9)]
            self.assertAlmostEqual(r[3], 3.0 * k / 9, places=12)
            self.assertAlmostEqual(r[4], 0.06 * k / 9, places=12)

    def test_components_add(self):
        a = mt.parse_component("step", "start=1,rise=0,trans=1:0:0")
        b = mt.parse_component("drift", "trans=0:0:2")
        rows = mt.synthesize(10, 2, 0.0, [a, b])
        self.assertEqual(rows[0][1:4], [0.0, 0.0, 0.0])
        self.assertEqual(rows[-1][1:4], [1.0, 0.0, 2.0])
        mid = rows[10]  # t = 1.0: the step is up, the drift half way
        self.assertAlmostEqual(mid[0], 1.0)
        self.assertEqual(mid[1], 1.0)
        self.assertAlmostEqual(mid[3], 1.0)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            mt.parse_component("nod", "start=1,rise=0.4,trans=0:0:1")  # no fall
        with self.assertRaises(ValueError):
            mt.parse_component("nod", "start=1,rise=-1,fall=1")
        with self.assertRaises(ValueError):
            mt.parse_component("step", "start=1,rise=0,trans=0:1")
        with self.assertRaises(ValueError):
            mt.parse_component("drift", "trans=0:0:1,shape=square")
        with self.assertRaises(ValueError):
            mt.parse_component("step", "start=1,rise=0,colour=red")
        with self.assertRaises(ValueError):
            mt.synthesize(0, 1, 0.0, [])
        # implementation review: one sample, a start that is not a number
        with self.assertRaises(ValueError):
            mt.synthesize(100, 0.001, 0.0, [])
        with self.assertRaises(ValueError):
            mt.synthesize(100, 1, float("nan"), [])


class Files(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def test_output_has_increasing_times_and_the_columns(self):
        out = os.path.join(self.d, "nod.tsv")
        mt.main(["--rate", "100", "--duration", "5", "--start", "-0.5", "--out", out,
                 "--nod", "start=1,rise=0.4,hold=1,fall=0.4,trans=0:2:3,rot=0.06:0:0"])
        header, rows = read(out)
        self.assertEqual(header, ["time"] + mt.COLS)
        self.assertEqual(len(rows), 501)
        self.assertEqual(rows[0][0], -0.5)
        self.assertTrue(all(b[0] > a[0] for a, b in zip(rows, rows[1:])))
        self.assertTrue(all(math.isfinite(x) for r in rows for x in r))

    def test_from_a_sidecar_round_trips(self):
        src = os.path.join(self.d, "physio_motion.tsv")
        with open(src, "w") as f:
            f.write("\t".join(mt.COLS) + "\n")
            for i in range(5):
                f.write("\t".join(["0", str(0.1 * i), "n/a", "0", "0", str(0.01 * i)]) + "\n")
        dump({"SamplingFrequency": 50.0, "StartTime": -0.02}, os.path.join(self.d, "physio_motion.json"))
        out = os.path.join(self.d, "timed.tsv")
        mt.main(["--from", src, "--out", out])
        header, rows = read(out)
        self.assertEqual(header[0], "time")
        self.assertEqual([r[0] for r in rows], [-0.02 + i / 50.0 for i in range(5)])
        self.assertEqual([r[2] for r in rows], [0.1 * i for i in range(5)])
        self.assertEqual([r[3] for r in rows], [0.0] * 5)

    def test_from_refuses_two_clocks_and_no_rate(self):
        src = os.path.join(self.d, "t.tsv")
        dump_text("time\t" + "\t".join(mt.COLS) + "\n0\t0\t0\t0\t0\t0\t0\n", src)
        dump({"SamplingFrequency": 10}, os.path.join(self.d, "t.json"))
        with self.assertRaises(ValueError):
            mt.from_sidecar(src)
        dump({"Columns": mt.COLS}, os.path.join(self.d, "t.json"))
        with self.assertRaises(ValueError):
            mt.from_sidecar(src)


if __name__ == "__main__":
    unittest.main()
