# aslscan

aslscan simulates arterial spin labeling (ASL) MRI data. You give it a description of an ASL scan
and a digital brain phantom. It produces a BIDS dataset that looks like real scanner output,
along with the ground-truth maps used to create it. Because the true perfusion values are known
exactly, the simulated data can be used to test and compare ASL processing pipelines.

aslscan is at an early stage (version 0.0.0). The scope described below is deliberately narrow,
and aslscan reports an error when a protocol asks for something it does not model. It does not
silently ignore those requests.

## What it does

ASL measures blood flow by magnetically "labeling" the water in arterial blood and imaging the
brain after the labeled blood arrives. A scan alternates *label* and *control* images. The small
difference between them is proportional to perfusion.

aslscan simulates this in three stages.

1. **Kinetics.** For each phantom voxel, the Buxton general kinetic model gives the size of the
   label-control difference at each image's timing. The inputs are the phantom's perfusion and
   arrival-time maps and the protocol's labeling duration and post-labeling delay.
2. **Signal.** The static tissue signal comes from each voxel's M0 and T1 values and the
   repetition time. The labeled blood signal is carried separately so that it can relax with
   its own T2.
3. **Acquisition.** The images pass through a model of a 2D spin-echo echo-planar (EPI) readout,
   provided by the companion library [mrsim-acq](https://github.com/PennLINC/mrsim-acq). This
   step can add distortion from a B0 fieldmap, signal decay during the readout, Gibbs ringing,
   partial Fourier, multiple receive coils, parallel imaging, ghosting, spikes, and noise.

Physics is evaluated at the phantom's resolution first and then averaged down to the scan
resolution. Partial-volume effects at tissue boundaries are therefore handled correctly.

The design and its validation against ASLDRO, the Python ASL simulator used as the reference,
are documented in
[the design spec](https://github.com/PennLINC/mrsim-acq/blob/main/docs/specs/2026-09-21-mrsim-acq-aslscan-design.md).

### Supported protocols

| Supported | Not yet supported (reported as an error) |
|---|---|
| PCASL, CASL, and PASL labeling (PASL requires bolus cut-off, e.g. QUIPSS II/Q2TIPS) | Background suppression |
| 2D multi-slice acquisitions, with slice timing applied to the delay | 3D readouts (GRASE, stack-of-spirals) |
| Spin-echo contrast | Gradient-echo and inversion-recovery contrast |
| Single or multiple post-labeling delays | Look-Locker, Hadamard, velocity-selective labeling |
| `control`, `label`, `deltam`, and `m0scan` volumes | `cbf` volumes; multi-echo (differing echo times) |
| M0 included in the series, as a separate scan, estimated, or absent | Vascular crushing; head motion |

Phase encoding must lie along the second image axis (`j` or `j-`).

## Installation

aslscan is written in Rust. It has been built and tested on Linux, including Windows Subsystem
for Linux (WSL).

1. Install the Rust toolchain with [rustup](https://rustup.rs/).
2. Clone aslscan and mrsim-acq **into the same parent directory**. aslscan finds mrsim-acq
   at `../mrsim-acq`.

   ```bash
   git clone https://github.com/PennLINC/mrsim-acq.git
   ```

   ```bash
   git clone https://github.com/PennLINC/aslscan.git
   ```

3. Build the command-line program from inside the `aslscan` directory:

   ```bash
   cargo build --release --features cli,kspace,par
   ```

   The program is written to `target/release/aslscan`. The three features enable the
   command-line interface (`cli`), the faster FFT-based acquisition code (`kspace`), and
   multi-threading (`par`). You can also run `cargo install --path . --features cli,kspace,par`
   to put `aslscan` on your `PATH`.

## Quick start

The repository includes a small cropped phantom and example protocols. This command simulates a
short PCASL series on them and runs in well under a second:

```bash
./target/release/aslscan --asl-json tests/fixtures/protocols/crop_pcasl/asl.json --aslcontext tests/fixtures/protocols/crop_pcasl/aslcontext.tsv --overlay tests/fixtures/protocols/crop_pcasl/overlay.toml --phantom tests/fixtures/phantom-crop --out work/demo
```

The cropped phantom is only 24 x 24 x 6 mm and is intended for testing. For realistic data, use a
full-brain phantom (see [Making a phantom](#making-a-phantom)).

## Usage

```
aslscan --asl-json <JSON> --aslcontext <TSV> --phantom <DIR> --out <DIR> [OPTIONS]
```

| Option | Meaning |
|---|---|
| `--asl-json` | BIDS ASL sidecar (`*_asl.json`) describing the scan protocol. |
| `--aslcontext` | BIDS `*_aslcontext.tsv` listing the volume types in order. One output volume is produced per row. |
| `--phantom` | Directory containing the phantom maps. |
| `--out`, `-o` | Directory to write the BIDS dataset to. |
| `--overlay` | Optional TOML file with settings BIDS does not record (see below). |
| `--t2-mode` | `auto` (default), `class`, or `voxel`. Controls how T2 and T2* are represented; see below. |
| `--sub`, `--ses` | Subject and session labels for the output file names. The subject defaults to `01`. |
| `--seed` | Random seed for noise and other random effects. Overrides the overlay's seed. |

Run `aslscan --help` for the full list.

### Protocol inputs

The protocol is read from a standard BIDS ASL sidecar and `aslcontext.tsv`, as found in a real
BIDS dataset. In practice, you can take these two files from a dataset whose acquisition you
want to reproduce. aslscan uses the labeling type, labeling duration, post-labeling delay,
bolus cut-off settings, M0 type, repetition time, echo time, total readout time, voxel size,
slice timing, and phase-encoding direction.

The following BIDS fields are required:

- `BackgroundSuppression`, which must be `false`
- `MRAcquisitionType: "2D"`, along with `SliceTiming`
- `PhaseEncodingDirection` and `TotalReadoutTime`

Many public datasets use background suppression and will be rejected for that reason. The
protocols under `tests/fixtures/protocols/` (`pcasl_single`, `pcasl_multipld`, `pasl_cutoff`,
`crop_pcasl`) are complete working examples.

### Overlay file

Some values needed for simulation are not part of BIDS, such as the blood-brain partition
coefficient, the number of receive coils, and the noise level. These can be set in an optional
TOML file. All keys are optional. Values are taken from the overlay first, then from the BIDS
sidecar or the phantom, and then from the defaults below. The output sidecar records the value
used for each setting and where it came from.

```toml
seed = 20260923                # random seed

[kinetic]
label_efficiency = 0.85        # default 0.85 (PCASL/CASL), 0.98 (PASL); BIDS LabelingEfficiency is used if present
lambda_blood_brain = 0.9       # ml/g; default 0.9
t1_arterial_blood = 1.65       # s; default 1.65 at 3 T, 1.35 at 1.5 T

[signal]
acq_contrast = "se"            # only "se" (spin echo) is supported
t2_blood = 0.165               # s; default 0.165 at 3 T, 0.290 at 1.5 T

[acquisition]
oversample = 2                 # in-plane simulation resolution relative to the scan (default 2)
matrix = [64, 64]              # in-plane scan matrix; default: phantom extent / voxel size
noise_variance = 0.0           # default 0: no noise
signal_scale = 100.0           # overall intensity scale (default 100)
n_coils = 1                    # receive coils (default 1)
acs_lines = 24                 # parallel-imaging calibration lines (default 24)
partial_fourier = 1.0          # fraction of phase-encode lines acquired (default 1.0)
pf_mode = "fiberfox"           # "fiberfox" or "contiguous"
window = "none"                # "none", "hann", "tukey:<alpha>", "fermi:<radius>,<width>"
t_inhom = 50.0                 # ms; fallback T2' where the phantom provides none
ghost_offset = 0.0             # Nyquist ghost strength (0 = off)
n_spikes = 0                   # k-space spikes per slice (0 = off)
spike_amplitude = 1.0
eddy_strength = 0.0            # eddy-current terms (0 = off)
eddy_quad = 0.0
eddy_phase = 0.0
eddy_tau = 70.0                # ms

[m0]
repetition_time = 8.0          # s; required when M0Type is "Separate"
```

Note that **noise is off by default**. Set `noise_variance` to add it. An unrecognized key is an
error, so a misspelled key is reported rather than ignored.

### Phantom

The phantom is a directory of NIfTI images on a common grid, each with a JSON sidecar giving its
`Units`:

| File | Contents | Units |
|---|---|---|
| `perfusion.nii.gz` | cerebral blood flow | `ml/100g/min` |
| `att.nii.gz` | arterial transit time | `s` |
| `T1map.nii.gz` | tissue T1 | `s` |
| `T2map.nii.gz` | tissue T2 | `s` |
| `T2starmap.nii.gz` | tissue T2* | `s` |
| `M0map.nii.gz` | equilibrium magnetization | `arbitrary` |
| `dseg.nii.gz` | tissue labels; `dseg.json` names them in a `LabelMap` | `label indices` |
| `fieldmap.nii.gz` | B0 fieldmap (optional) | `Hz` |

An optional `phantom.json` can supply `LambdaBloodBrain`, `T1ArterialBlood`, and
`MagneticFieldStrength`. The field strength must match the protocol's. Without a fieldmap, no EPI
distortion is simulated.

`--t2-mode` controls how T2 and T2* are handled. In `class` mode, each tissue label has one
uniform T2 and T2*. This mode is exact and fast, but requires the maps to be constant within each
label. `voxel` mode allows the values to vary from voxel to voxel. `auto` chooses `class` when the
phantom permits it.

### Making a phantom

`tools/hrgt_to_bids.py` converts the ground-truth phantoms distributed with ASLDRO v2.2.0 into
the layout above. It requires a Python environment with ASLDRO installed.

```bash
python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t --out phantom-3t
```

`hrgt_icbm_2009a_nls_1.5t` is the 1.5 T version. Add `--crop x0:x1 y0:y1 z0:z1` to extract a
smaller region.

## Output

```
<out>/
  dataset_description.json
  README
  .bidsignore
  sub-01/perf/
    sub-01_part-mag_asl.nii.gz    + .json    magnitude images, one volume per aslcontext row
    sub-01_part-phase_asl.nii.gz  + .json    phase images (radians)
    sub-01_aslcontext.tsv
    sub-01_m0scan.nii.gz          + .json    only when M0Type is "Separate"
    ground-truth/
      sub-01_desc-perfusion_gt.nii.gz        true values on the scan grid
      sub-01_desc-att_gt.nii.gz
      sub-01_desc-T1map_gt.nii.gz, _desc-T2map_gt, _desc-M0map_gt
      sub-01_desc-dseg_gt.nii.gz             tissue labels
      sub-01_desc-deltam_gt.nii.gz           noise-free label-control difference, per volume
```

The dataset passes the BIDS validator. The `ground-truth/` directory is listed in `.bidsignore`.
Each `*_asl.json` sidecar repeats the input protocol and adds an `AslscanSimulation` section.
That section records every value aslscan resolved, including grid sizes, kinetic constants and
their sources, acquisition settings, and random seeds. When a setting in the simulation
overrides a value in the input sidecar, the standard BIDS field reports the value that was
simulated, and the original input value is kept under
`AslscanSimulation.InputValuesReplaced`.

## Testing

```bash
cargo test --features io,test-hooks
```

This runs the unit tests and the end-to-end tests, using the fixtures under `tests/fixtures/`.
The reference values for the kinetic and signal models were generated with ASLDRO by
`tools/gen_gkm_fixtures.py` and `tools/gen_mrsignal_fixtures.py`. The `test-hooks` feature exists
only for these tests. Do not enable it when producing data.

## License

MIT or Apache-2.0, at your option.
