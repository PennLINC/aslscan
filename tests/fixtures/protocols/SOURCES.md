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
