# Protocol fixture sources

Fetched from https://github.com/bids-standard/bids-examples at commit `7150efcf2c040465e9fcb6745f0b5aa4e84919c1` by tools/fetch_protocol_fixtures.sh.
Only the JSON sidecar and aslcontext.tsv are kept; no image data.

- `asl005/`: `asl005/sub-Sub103/perf/sub-Sub103_asl.json` and `..._aslcontext.tsv`
- `asl004/`: `asl004/sub-Sub1/perf/sub-Sub1_asl.json` and `..._aslcontext.tsv`
- `asl001/`: `asl001/sub-Sub103/perf/sub-Sub103_asl.json` and `..._aslcontext.tsv`
- `asl003/`: `asl003/sub-Sub1/perf/sub-Sub1_asl.json` and `..._aslcontext.tsv`
- `asl002/`: `asl002/sub-Sub103/perf/sub-Sub103_asl.json` and `..._aslcontext.tsv`

## P5 acceptance cases

- `asl005_p5/overlay.toml`: an overlay for the unchanged `asl005/` sidecar and context (P5 plan,
  Task 10): matrix, segmentation, phase-encode direction, M0 repetition time.
- `asl003_p5/`: a derivative of `asl003/`, with its overlay. Changed: the first four volumes
  (PostLabelingDelay 0.3, 0.3, 0.6, 0.6 s) are removed from `aslcontext.tsv` and from the
  `PostLabelingDelay` array, because they read before the 0.7 s bolus cutoff, which aslscan
  refuses for PASL (`protocol.rs`, the PASL cutoff rule; lifting it is not P5 work). Nothing else
  changes: `RepetitionTimePreparation` is a scalar, and `EffectiveEchoSpacing`, `NumberShots`,
  `FlipAngle` and the suppression pulse times are kept.
- `asl001_p5/overlay.toml`: an overlay for the unchanged `asl001/` sidecar and context (P5 plan,
  Task 14): the in-plane matrix and the spiral's interleaves, readout time and dwell time, which
  the sidecar does not carry and which have no defensible default.

## P6 acceptance cases

- `p6_multite/`: written for aslscan (P6 plan, Task 6), not fetched. PCASL on the crop with
  exchange, a gradient-echo readout at 60 degrees read at three echo times: one sidecar per echo
  (`asl-echo-{1,2,3}.json`, identical except `EchoTime` 13, 32, 51 ms; pass them as repeated
  `--asl-json`), one `aslcontext.tsv`, the overlay.
- `p6_multite_se/`: its spin-echo variant at 15, 30, 45 ms, the 2 ms refocusing reserve fitting
  between the 12 ms EPI blocks.
- `p6_hadamard/`: written for aslscan (P6 plan, Task 11). H8 PCASL on the crop: seven sub-boli of
  0.25 s, PLD 1.5 s, each row's PostLabelingDelay its sub-bolus's effective delay; two encoding
  cycles and an m0scan row before the first (the context lists the decoded volumes); exchange on.
- `p6_hadamard_3t/`: the same on the 3 T phantom cropped to 97 mm in z (`work/phantom-3t-z97`, 24
  slices of 4.05 mm), 3.5 mm in plane, slices 35 ms apart, TR 4.5 s.
- `p6_hadamard_grase/`: the GRASE variant on `work/phantom-3t-z120` with asl005_p5's readout (64 x 64
  in four ky segments, four shots per raw volume, EchoTime 13.28 ms, DwellTime 3.2 us, 130 degrees).
- `p6_hadamard_multite/`: Hadamard x multi-TE on the crop: one sidecar per echo (gradient echo at
  13, 32, 51 ms, 60 degrees, slices 60 ms apart).
- `p6_ll/`: written for aslscan (P6 plan, Task 15). Look-Locker PASL on the crop (Q2TIPS, 0.7 s cutoff),
  2D EPI: a control cycle and a label cycle of twelve gradient-echo readouts at 35 degrees, 0.3 s
  apart from PLD 0.8 s, TR 4.5 s, a separate M0 at the series' flip.
- `p7_quasar/`: written for aslscan (P7 plan, Task 5) by `tools/make_p7_quasar.py`, not fetched. QUASAR-like
  Look-Locker PASL on the crop (Q2TIPS, 0.7 s cutoff), 2D EPI: control and label cycles of thirteen
  gradient-echo readouts at 35 degrees, 0.3 s apart from 0.8 s, TR 4.8 s, uncrushed then crushed at 4 cm/s
  (`VascularCrushingVENC` per readout); exchange, the arterial compartment with per-label arterial
  velocities, a separate M0.
- `p7_ll_multite/`: written for aslscan (P7 plan, Task 6). `p6_ll` read at three echo times, 12, 26 and
  40 ms (`asl-echo-{1,2,3}.json`, identical except `EchoTime`; pass them as repeated `--asl-json`), the
  third echo's block ending inside the 50 ms between slice groups.
- `p7_ll_hadamard/`: written for aslscan (P7 plan, Task 8) by `tools/make_p7_ll_hadamard.py`. H8 PCASL on the crop,
  each encoded preparation read by four Look-Locker readouts: seven sub-boli of 0.25 s, PLD_n = 0.5, 0.8, 1.1,
  1.4 s, 35 degrees, TR 4.5 s, two encoding cycles after an included M0, exchange on; the context lists the 56
  decoded volumes readout-major.
- `p7_ge3d/`: written for aslscan (P7 plan, Task 15). `p5_grase`'s protocol (PCASL on the crop, two shots in two ky
  segments, suppression, exchange, physiological noise, a separate M0) read by a 3D gradient-echo stack of EPI
  (`PulseSequenceType` "3D EPI", `FlipAngle` 12, the excitation), 40 ms between excitations, the label entering
  the slab 0.5 s after labeling.
- `p7_ll3d/`: written for aslscan (P7 plan, Task 16) by `tools/make_p7_ll3d.py`. 3D Look-Locker PASL on the crop
  (Q2TIPS, 0.7 s): cycles of six readouts 0.3 s apart from 0.8 s, each a sub-train of the 3D EPI train (8
  degrees, 40 ms between excitations), two shots per group, TR 4.0 s, the label entering the slab 0.3 s after
  labeling, a separate M0.
