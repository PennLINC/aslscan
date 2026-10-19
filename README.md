# aslscan

aslscan simulates arterial spin labeling (ASL) MRI data. You give it a description of an ASL scan
and a digital brain phantom. It produces a BIDS dataset that looks like real scanner output,
along with the ground-truth maps used to create it. Because the true perfusion values are known
exactly, the simulated data can be used to test and compare ASL processing pipelines.

aslscan is at an early stage (version 0.0.0). It reports an error when a protocol asks for
something it does not model, and never silently ignores the request.

## What it does

ASL measures blood flow by magnetically "labeling" the water in arterial blood and imaging the
brain after the labeled blood arrives. A scan alternates *labeled* and *control* images. The small
difference between them, the ASL difference image ΔM (control minus labeled), is proportional to
perfusion.

aslscan simulates this in three stages.

1. **Kinetics.** For each phantom voxel, the Buxton general kinetic model gives ΔM at each image's
   timing. It uses the phantom's cerebral blood flow (CBF) and arterial transit time (ATT) maps and
   the protocol's labeling duration (LD) and post-labeling delay (PLD), or for PASL its inversion
   time (TI) and bolus duration (BD). Optional
   extensions add the following:
   - water exchange between blood and tissue;
   - a separate arterial (macrovascular) compartment, which vascular crushing can remove;
   - the effect of background-suppression pulses on each part of the labeled bolus;
   - cardiac, respiratory and drift fluctuations.
2. **Signal.** The static tissue signal comes from each voxel's M0 and T1 values, the repetition
   time and the contrast. The contrast can be spin echo, inversion recovery or gradient echo. Under
   background suppression, aslscan follows the tissue magnetization through every pulse. The labeled
   blood is carried in its own compartments so that it relaxes with its own T2.
3. **Acquisition.** The images pass through a model of the scanner readout, provided by the
   companion library [mrsim-acq](https://github.com/PennLINC/mrsim-acq). The readout can be 2D
   echo-planar (EPI), 3D GRASE, 3D stack-of-spirals, or a 3D gradient-echo EPI train. This step can add
   the following:
   - distortion from a B0 fieldmap;
   - signal decay during the readout;
   - Gibbs ringing;
   - partial Fourier;
   - multiple receive coils and parallel imaging;
   - ghosting and spikes;
   - noise;
   - head motion.

Physics is evaluated at the phantom's resolution first and then averaged down to the scan
resolution, so partial-volume effects at tissue boundaries are handled correctly.

The design and its validation against ASLDRO, the Python ASL simulator used as the reference,
are documented in the [mrsim-acq specs](https://github.com/PennLINC/mrsim-acq/tree/main/docs/specs).
The first spec covers the core. The later addenda cover ASLDRO compatibility (P2); motion,
background suppression and inversion recovery (P3); the vascular and physiological extensions
(P4); 3D readouts and gradient echo (P5); Hadamard, Look-Locker and multi-echo (P6); and QUASAR, the
Look-Locker combinations and the 3D gradient-echo EPI train (P7).

### Terminology

aslscan uses the terms of the OSIPI ASL lexicon ([Suzuki et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10950547/)),
and in code the same name for the same model term throughout:

| Term | In aslscan |
|---|---|
| Labeling duration (LD, τ), (P)CASL | `tau`; BIDS `LabelingDuration` |
| Bolus duration (BD), PASL | `tau`; BIDS `BolusCutOffDelayTime` (its first value, TI1, for Q2TIPS) |
| Post-labeling delay (PLD), (P)CASL; inversion time (TI), PASL | BIDS `PostLabelingDelay` (BIDS uses it for both) |
| Inflow time: PLD + LD for (P)CASL, TI for PASL | `t`, the general kinetic model's clock |
| Arterial transit time (ATT) | `att`; phantom `att.nii.gz` |
| Labeling efficiency (α) | `alpha`; overlay `labeling_efficiency` |
| Blood-brain partition coefficient (λ), mL/g | `lambda`; overlay `lambda_blood_brain` |
| T1 of arterial blood (T1b) | `t1b`; overlay `t1_arterial_blood` |
| Equilibrium magnetization of tissue (M0t) and of blood (M0b = M0t / λ) | `m0`, `m0b`; phantom `M0map.nii.gz` |
| ASL difference image (ΔM), control minus labeled | `delta_m`; `aslcontext.tsv` `deltam` |
| Cerebral blood flow (CBF), mL/100 g/min | phantom `perfusion.nii.gz` |

Two departures from the lexicon:
- **Labeling efficiency.** OSIPI's α includes the label lost to background suppression. aslscan's `alpha`
  is the labeling pulse's efficiency alone, because the background-suppression pulses are simulated
  one by one.
- **The arterial compartment's ATT.** The lexicon has no term for the macrovascular extension, and
  its ATT (with the aliases AAT and BAT) is the arrival in the tissue. The arrival of label in a
  voxel's arterial compartment is therefore called the arterial compartment's ATT: overlay
  `arterial_compartment_att`, phantom `att_arterial.nii.gz`, code `att_arterial`. Its partner is
  the arterial blood volume (aBV, `abv`).

### Supported protocols

| Area | Supported |
|---|---|
| Labeling | PCASL, CASL, PASL (with bolus cut-off, e.g. QUIPSS II/Q2TIPS); Hadamard time-encoded (P)CASL |
| Delays | Single or multiple post-labeling delays; Look-Locker readouts (several per labeling; 2D EPI or 3D EPI) |
| Readout | 2D multi-slice EPI (with multiband), 3D GRASE, 3D stack-of-spirals (with partial Fourier and acceleration along kz), 3D gradient-echo EPI (stack of EPI) |
| Contrast | Spin echo, inversion recovery, gradient echo |
| Echoes | One, or several echo times (2D EPI or 3D EPI; the BIDS `echo-N` layout) |
| Combinations | Look-Locker with every vascular extension (QUASAR), with Hadamard, and with several echo times |
| Background suppression | Pulses at any times; a global-bolus model or a per-parcel bolus-position model |
| Volumes | `control`, `label`, `deltam`, `m0scan`; M0 included, separate, estimated, or absent |
| Vascular | Water exchange, an arterial compartment, vascular crushing |
| Noise and motion | Thermal noise, physiological fluctuations, rigid head motion, within-volume (shot) motion |

Not supported, each reported as an error:
- velocity-selective labeling, which has no BIDS labeling type yet;
- Look-Locker or several echo times with GRASE or spiral readouts;
- partial Fourier or acceleration along kz with GRASE or the 3D EPI train, or with Hadamard encoding;
- the 3D gradient-echo EPI train with a smooth (not label-wise) T1 map;
- `cbf` volumes;
- phase encoding along any axis other than the second (`j` or `j-`).

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
   command-line interface (`cli`), the faster FFT-based acquisition code (`kspace`, required for
   spiral readouts), and multi-threading (`par`). You can also run
   `cargo install --path . --features cli,kspace,par` to put `aslscan` on your `PATH`.

## Quick start

The repository includes a small cropped phantom and example protocols. This command simulates a
short PCASL series on them and runs in well under a second:

```bash
./target/release/aslscan --asl-json tests/fixtures/protocols/crop_pcasl/asl.json --aslcontext tests/fixtures/protocols/crop_pcasl/aslcontext.tsv --overlay tests/fixtures/protocols/crop_pcasl/overlay.toml --phantom tests/fixtures/phantom-crop --out work/demo
```

The cropped phantom is only 24 x 24 x 6 mm and is intended for testing. For realistic data, use a
full-brain phantom (see [Making a phantom](#making-a-phantom)).

Every directory under `tests/fixtures/protocols/` is a working example; `SOURCES.md` there
describes each one:

| Fixture | Shows |
|---|---|
| `crop_pcasl`, `pcasl_single`, `pcasl_multipld`, `pasl_cutoff` | Basic PCASL and PASL, single and multiple delays |
| `p4_all` | Exchange, the arterial compartment, crushing, bolus-position suppression, physiology |
| `p5_ge`, `p5_grase` | Gradient echo with suppression; a small 3D GRASE protocol |
| `asl001`-`asl005` (with the `_p5` overlays) | Real BIDS examples, including 3D spiral (asl001) and GRASE (asl005) |
| `p6_multite`, `p6_multite_se` | Three echo times (one sidecar per echo) |
| `p6_hadamard`, `p6_hadamard_grase`, `p6_hadamard_multite` | Hadamard-encoded PCASL in 2D, in 3D, and with three echoes |
| `p6_ll` | Look-Locker PASL, twelve readouts per labeling |
| `p7_quasar` | QUASAR-like Look-Locker PASL: crushed and uncrushed cycles, exchange, the arterial compartment |
| `p7_ll_multite`, `p7_ll_hadamard` | Look-Locker with three echo times; Look-Locker with Hadamard (H8, four readouts) |
| `p7_ge3d`, `p7_ll3d`, `p7_multite3d` | The 3D gradient-echo EPI train: plain, Look-Locker, three echoes per excitation |
| `p8_spiral_kz` | An accelerated stack-of-spirals pCASL: 6/8 partial Fourier and acceleration 2 along kz (needs a full phantom) |

## Usage

```
aslscan --asl-json <JSON> --aslcontext <TSV> --phantom <DIR> --out <DIR> [OPTIONS]
```

| Option | Meaning |
|---|---|
| `--asl-json` | BIDS ASL sidecar (`*_asl.json`) describing the scan protocol. Repeat it once per echo, in echo order, for a multi-echo series. |
| `--aslcontext` | BIDS `*_aslcontext.tsv` listing the volume types in order. One output volume is produced per row. |
| `--phantom` | Directory containing the phantom maps. |
| `--out`, `-o` | Directory to write the BIDS dataset to. |
| `--overlay` | Optional TOML file with settings BIDS does not record (see below). |
| `--t2-mode` | `auto` (default), `class`, or `voxel`. Controls how T2 and T2* are represented; see below. |
| `--sub`, `--ses` | Subject and session labels for the output file names. The subject defaults to `01`. |
| `--seed` | Random seed for noise and other random effects. Overrides the overlay's seed. |
| `--compat-asldro` | Restrict the simulation to what ASLDRO can express, for direct comparison with it (see below). |

Run `aslscan --help` for the full list.

### Protocol inputs

The protocol is read from a standard BIDS ASL sidecar and `aslcontext.tsv`, as found in a real
BIDS dataset. In practice, you can take these two files from a dataset whose acquisition you
want to reproduce. aslscan reads, among others:
- labeling: type, labeling duration, post-labeling delay (for PASL, the TI), bolus cut-off, M0 type;
- timing: repetition time, echo time, total readout time, slice timing;
- geometry: voxel size, phase-encoding direction, flip angle;
- background suppression and its pulse times;
- vascular crushing and its VENC;
- for 3D readouts, `PulseSequenceType` and `NumberShots`.

What each acquisition type needs:
- **2D:** `MRAcquisitionType: "2D"`, `SliceTiming`, `PhaseEncodingDirection` and
  `TotalReadoutTime`.
- **3D:** `MRAcquisitionType: "3D"` with `PulseSequenceType` naming GRASE, spiral or EPI (for example
  `"3D EPI"`). For details that BIDS does not carry (segmentation, spiral interleaves and readout time,
  the gradient-echo train's excitation spacing), use the overlay's `[readout]` table.

### Overlay file

Some values needed for simulation are not part of BIDS, such as the blood-brain partition
coefficient, the number of receive coils, and the noise level. These can be set in an optional
TOML file. All keys are optional unless a feature requires them. Values are taken from the
overlay first, then from the BIDS sidecar or the phantom, and then from the defaults. The output
sidecar records the value used for each setting and where it came from. An unrecognized key is an
error, and so is a key belonging to a feature that is off. Either way, a mistake is reported rather
than ignored.

```toml
seed = 20260923                # random seed

[kinetic]
labeling_efficiency = 0.85     # alpha; else BIDS LabelingEfficiency, else 0.85 (PCASL), 0.68 (CASL), 0.98 (PASL)
lambda_blood_brain = 0.9       # lambda, mL/g; default 0.9
t1_arterial_blood = 1.65       # T1b, s; default 1.65 at 3 T, 1.35 at 1.5 T
exchange_time = 0.5            # s; turns on blood-to-tissue water exchange (off by default)

[signal]
acq_contrast = "se"            # "se" (spin echo, default), "ir" (inversion recovery), "ge" (gradient echo)
t2_blood = 0.165               # s; default 0.165 at 3 T, 0.290 at 1.5 T
t2_arterial = 0.165            # s; the arterial compartment's T2 (default: the blood T2)
inversion_time = 1.0           # s; "ir" only (default: sidecar InversionTime, else 1.0)
excitation_flip_angle = 60.0   # degrees; "ir" and "ge" (default: sidecar FlipAngle, else 90)
inversion_flip_angle = 180.0   # degrees; "ir" only

[acquisition]
oversample = 2                 # in-plane simulation resolution relative to the scan (default 2)
matrix = [64, 64]              # in-plane scan matrix; default: phantom extent / voxel size
noise_variance = 0.0           # default 0: no noise
signal_scale = 100.0           # overall intensity scale (default 100)
n_coils = 1                    # receive coils (default 1)
coil_rows = 1                  # rings of coils along z (default 1); spiral readouts only
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
flip_angle = 30.0              # degrees; a Look-Locker series' separate M0 (see below)
```

Note that **noise is off by default**. Set `noise_variance` to add it.

The remaining tables switch on optional features.

**Background suppression** (with `BackgroundSuppression: true` in the sidecar):

```toml
[background_suppression]
inversion_efficiency = 0.95    # fraction of magnetization each pulse inverts (default 0.95)
presaturation = false          # a saturation pulse at labeling start
pulse_times_per_pld = [[1.42], [1.92]]  # multi-delay series: one pulse list per delay
model = "global-bolus"         # or "bolus-position": each part of the bolus sees only the pulses after it
pulse_region = "slab"          # "global" or "slab"; bolus-position only
slab_entry_time = 0.3          # s, or "arrival"; with "slab"
```

**Motion:**

```toml
[motion]
mode = "random"                # "off", "random", "linear", or "trajectory"
trans_mm = [2.0, 2.0, 1.0]     # random/linear amplitudes
rot_deg = [1.0, 1.0, 2.0]
volumes = [5, 20, 40]          # volumes affected (default all)
trajectory = "motion.tsv"      # "trajectory" mode: trans_x/y/z (mm), rot_x/y/z (radians), one row per volume
within_volume_sampling = "volume"  # 2D: "volume" (each shot has its volume's pose) or "slice" (the pose at each shot's time)

[motion.within_volume]         # shot events: 2D slice groups (single-band or multiband), or segmented-3D shots
dropout_rate = 0.1
severity = 0.5
jump_mm = [0.5, 0.0, 0.0]
jump_deg = [0.0, 0.0, 1.0]
# events = "events.tsv"        # 2D: explicit events in the desc-motionEvents_gt.tsv format, instead of the draw
```

On a 2D readout, a shot is the set of slices excited together, taken from `SliceTiming` (any slice
order). With `within_volume_sampling = "slice"` (`linear` or `trajectory` mode), the volumes' poses
become a trajectory, placed at each volume's first shot and sampled at every shot's time. A shot's
transform is the volume's (or trajectory's) pose followed by the jumps of the events up to that
shot. Single-band events, slice sampling, an events file, or a multiband slice order that the
older path cannot represent move the images one shot at a time and write
`desc-motionShots_gt.tsv` (every shot's time and pose). Other motion requests take the per-volume
path, whose outputs are unchanged. A run's own `desc-motionEvents_gt.tsv` can be passed back as
`events` to reproduce it.

A slow drift moves each volume a little and leaves it otherwise clean; within-volume motion
staggers the slices of a volume.

**Spin history** (`[motion] spin_history = true`, 2D): tissue carried through the slice direction
into a slice not yet excited is excited twice in one repetition and is dark the second time;
tissue carried out of one is skipped and is bright in the next repetition. With the key set,
aslscan steps each phantom voxel's longitudinal magnetization through every slice excitation at
its shot's pose (a box slice profile as thick as the slice spacing) and adds the difference from
the closed form to the tissue images, so a head that does not move through the slices is
unchanged. It is written as `desc-spinHistory_gt` and described in `AslscanSimulation.Motion.SpinHistory`.
It needs a spin-echo series of control and label rows that share one preparation, with one echo,
no Look-Locker, Hadamard or ASLDRO compatibility; anything else is refused. Labeled blood is
treated as fresh.

What to expect, for 5 mm slices, TR 4.6 s and grey matter (T1 1.33 s):

| Motion | Tissue signal in the affected slice |
|---|---|
| In-plane, any size | unchanged |
| 0.5 mm through-plane between volumes | about 0.01% |
| 0.5 mm through-plane jump between two slices 40 ms apart | -9.7% |
| The same jump, neighbouring slice excited 0.4 s earlier | -7.3% |
| Tissue skipped for one volume | +3.1% on the skipped part |

Presaturation at labeling start erases what earlier volumes left, not the dip of a jump within a
volume. A slow drift therefore stays visually clean; the effect is a dip of a few percent in one
or two slices at an abrupt through-plane movement.

**Vascular and physiological extensions:**

```toml
[macrovascular]                # an arterial compartment; per label, or the phantom's abv/att_arterial maps
arterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }       # aBV, fraction
arterial_compartment_att = { grey_matter = 1.5, white_matter = 1.7, csf = 0.0 }      # s; the arterial compartment's ATT

[vascular_crushing]            # with VascularCrushing: true in the sidecar
arterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }  # cm/s

[physio]                       # fractional amplitudes; all default 0
tissue_cardiac = 0.02
tissue_respiratory = 0.01
tissue_drift = 0.01
label_cardiac = 0.03
label_respiratory = 0.0
label_drift = 0.0
cardiac_frequency = 1.0        # Hz (default 1.0)
respiratory_frequency = 0.25   # Hz (default 0.25)
```

**3D readouts:**

```toml
[readout]
type = "grase"                 # or "spiral"; overrides PulseSequenceType
ky_segments = 4                # GRASE
kz_segments = 1
interleaves = 8                # spiral (required)
spiral_readout_time = 4.0      # ms; spiral (required)
dwell_time = 4e-6              # s; spiral (otherwise the sidecar's DwellTime)
refocusing_flip_angle = 150.0  # degrees (default: the sidecar's FlipAngle, else 180)
kz_partial_fourier = 0.75      # spiral: fraction of the kz partitions read (default: the sidecar's, else 1)
kz_acceleration = 2            # spiral: every second partition is read (default: the sidecar's, else 1)
kz_partial_fourier_reconstruction = "zero-fill"  # spiral: or "pocs"
```

The 3D gradient-echo EPI train (`type = "epi3d"`, or a `PulseSequenceType` containing "EPI") has its
own keys, described under [3D gradient-echo EPI](#3d-gradient-echo-epi):

```toml
[readout]
type = "epi3d"
excitation_spacing = 40.0      # ms between the train's excitations (required)
slab_entry_time = 0.5          # s after labeling that the label enters the slab, or "arrival"
                               # (required, unless [background_suppression] gives it)
kz_segments = 2                # partitions split over shots; ky segments are NumberShots / kz_segments
kz_order = "centric"           # or "linear"
excitation_time = 2.0          # ms reserved for each excitation pulse (default 2)
node_tolerance = 1e-4          # the label interpolation's tolerance, relative to its peak (default 1e-4)
max_t1_groups = 16             # distinct tissue T1 values allowed (default 16)

[images]
max_memory_gib = 4.0           # limit on the compartment images held at once (default 4)
```

The P6 features have their own tables, described in the next section.

### Multi-echo, Hadamard and Look-Locker

**Several echo times.** Give one sidecar per echo with repeated `--asl-json`, in echo order, and
one `aslcontext.tsv`.
- Each sidecar carries its own scalar `EchoTime`, and the echo times must increase.
- Every other field must be the same in all the sidecars.
- The echoes of each excitation share the same longitudinal state. Each compartment decays with its
  own T2 (spin echo) or T2* (gradient echo).
- aslscan checks that every echo's readout, and every spin-echo refocusing pulse, fits between
  excitations.

```toml
[multi_te]
refocusing_time = 2.0          # ms reserved for each spin-echo refocusing pulse (default 2)
```

**Hadamard time-encoded labeling.** Add a `[hadamard]` table. The `aslcontext.tsv` lists the
*decoded* volumes:
- `deltam` rows, `H - 1` per encoding cycle, in sub-bolus order;
- `m0scan` rows, allowed only between cycles.

The sidecar's arrays describe those decoded volumes:
- each row's `LabelingDuration` is its sub-bolus's duration;
- each row's `PostLabelingDelay` is its sub-bolus's effective delay, from the end of that sub-bolus
  to the excitation.

aslscan simulates the encoded acquisition and decodes it. Its tissue-leakage report shows how much
static tissue signal survived decoding into each sub-bolus; this is non-zero when the tissue
differs between encoded volumes, for example after a gradient-echo transient.

```toml
[hadamard]
order = 8                      # 4, 8, 16 or 32 (required)
report_leakage = true          # also decode the tissue alone and report its leakage (default true)
```

**Look-Locker readouts.** Set `LookLocker: true` in the sidecar, with gradient echo
(`acq_contrast = "ge"`) and a 2D EPI or 3D gradient-echo EPI readout.
- `FlipAngle` is required: a scalar, or one value per volume.
- Each run of consecutive volumes of one type with increasing `PostLabelingDelay` is one labeling
  followed by several readouts.
- Each readout depletes both the tissue magnetization and the label that has already arrived.
- One readout per labeling with a single scalar flip reproduces the ordinary gradient-echo series
  exactly.
- With a `FlipAngle` array and a separate M0, give the M0's flip in `[m0] flip_angle`.

```toml
[look_locker]
readouts_per_cycle = 12        # optional check against the grouping the arrays give
```

**Look-Locker combinations.**
- **QUASAR.** Every vascular extension works under Look-Locker: water exchange, the arterial
  compartment, vascular crushing with a VENC per volume (for example alternating crushed and
  uncrushed cycles), and bolus-position suppression. Each readout depletes the intravascular,
  extravascular and arterial label it reads. In 2D the arterial blood is taken to be fresh at each
  readout, because it crosses a slice much faster than the readouts are spaced. The sidecar states
  this assumption.
- **With several echo times.** Give one sidecar per echo, as above. The echoes of a readout share its
  depletion.
- **With Hadamard.** The `aslcontext.tsv` lists the decoded volumes readout by readout: for each
  readout, its `H - 1` sub-boli in order, each row's `PostLabelingDelay` that sub-bolus's effective
  delay to the readout. `FlipAngle` and VENC must agree across the sub-boli of one readout. Each
  readout is decoded on its own.

### Stack of spirals: undersampling along kz

A stack of spirals can skip kz partitions in two ways, which shorten its echo train (the echo train
length is the number of partitions read, divided by `kz_segments`).

- **Partial Fourier** (`kz_partial_fourier`, or the sidecar's `PartialFourier` with
  `PartialFourierDirection: "k"`). The train reads the `round(fraction x partitions)` highest
  partitions: the kz centre, all of the positive side, and part of the negative side. The partitions
  left out are zero-filled by default, or reconstructed by POCS (below). The fraction must be above
  0.5.
- **Acceleration** (`kz_acceleration`, or the sidecar's `ParallelReductionFactorOutOfPlane`). The
  train reads every R-th partition, counted from the kz centre. The missing partitions are filled in
  from their nearest acquired neighbours in every coil (a GRAPPA-like kernel with four source
  partitions and no in-plane extent, after Vidorreta et al. 2017).

Acceleration needs receive coils whose sensitivities differ along z:

```toml
[acquisition]
n_coils = 8                    # must be a multiple of coil_rows
coil_rows = 4                  # at least kz_acceleration
```

The coils form `coil_rows` rings along z, each of `n_coils / coil_rows` elements.

The kernel is fitted once per series, on a calibration acquisition that stands in for the sequence's
preparation repetitions:
- it reads the first control or label volume of the series (failing that its first `m0scan`, then its
  first `deltam`), before any motion, without noise sources other than
  thermal noise;
- it uses the same train without the acceleration, which is longer and must also fit in the
  repetition time;
- its noise is drawn from the series seed.

Because the kernel does not depend on the volume being reconstructed, the reconstruction stays linear
within a series.

**POCS** (`kz_partial_fourier_reconstruction = "pocs"`) reconstructs the partitions that partial
Fourier leaves out, as the paper's sequence does, instead of leaving them zero:
- it works per coil along z, after the in-plane reconstruction and before the coils are combined;
- it takes the image's phase from a low-resolution image of the acquired centre partitions, then
  repeats ten times: give the image that phase, and put the acquired partitions back;
- it recovers through-plane resolution that zero filling loses. On a test object, zero filling was
  off by 20% of the image peak and POCS by 2%.

POCS is not linear: the reconstruction of a sum of two objects is not the sum of their
reconstructions. Two things follow.
- A `deltam` row is refused with POCS. aslscan simulates a `deltam` volume as the image of the
  label difference alone, which matches control minus label only for a linear reconstruction. List
  `control` and `label` rows and subtract the volumes yourself, as a scanner would.
- With POCS, each coil is reconstructed on its own, so run time grows with `n_coils`.

What is not modeled:
- homodyne partial-Fourier reconstruction;
- 2D (CAIPI) or in-plane acceleration of a spiral;
- undersampling along kz with GRASE, the 3D EPI train, or Hadamard encoding.

On a small test object, the kernel alone (against the same partitions without the acceleration)
was wrong by 1% to 3% of the image peak without noise. Run time
grows with `n_coils / coil_rows`, because coils in different rings share their in-plane computation.

### 3D gradient-echo EPI

A 3D gradient-echo EPI acquisition (a "stack of EPI") excites the whole slab once per partition. A
train of small-flip excitations is each followed by an EPI readout of one partition's k-space plane,
or one ky segment of it in a segmented acquisition. aslscan simulates every excitation of the train.
- `FlipAngle` is the excitation, in (0, 90] degrees: a scalar, or one value per volume under Look-Locker. `EffectiveEchoSpacing` or `TotalReadoutTime`
  gives the line spacing (the effective spacing times the number of ky segments). `NumberShots` is
  the number of trains per volume: `kz_segments` times the ky segments.
- `EchoTime` is the time of each partition readout's k-space centre. aslscan checks that every
  echo's readout fits between the excitation pulses and that the train fits in the repetition time.
  A large field of view needs a short effective echo spacing: each echo's readout lasts the number
  of phase-encoding lines times the effective spacing, however the shots are segmented.
- The tissue magnetization is followed exactly through every excitation, for each distinct tissue
  T1. A smooth T1 map has too many distinct values and is refused (see `max_t1_groups`).
- **Depletion from slab entry.** A 3D excitation covers the whole slab, including its feeding
  arteries. Label is therefore depleted by every excitation after it enters the slab, not only after
  it reaches its voxel. `slab_entry_time` is when the label enters the slab, in seconds after
  labeling, with one value for all arteries. `"arrival"` turns off depletion before the voxel.
- The label is read at each excitation, interpolated between node excitations chosen to meet
  `node_tolerance`. All voxels share the same nodes. On a realistic phantom, whose ATTs vary
  from voxel to voxel, nearly every excitation becomes a node. The result is then exact, but memory
  and run time grow with the length of the train.
- **Memory.** The sidecar records the estimated size of the images held at once, and a series over
  `max_memory_gib` is refused before it runs. The estimate counts only these images; the whole
  process uses about 1.3 to 1.5 times as much. A train with fewer excitations (more `kz_segments`)
  needs fewer nodes.
- The ground truth (`desc-deltam_gt`) is the label at the excitation that reads the centre of
  k-space, depleted from slab entry.
- Look-Locker, several echo times per excitation, Hadamard, the vascular extensions, physiology, and
  motion (per volume and per shot) all work with this readout.

### Comparing with ASLDRO

`--compat-asldro` (or `[compat] asldro = true`) pins the simulation to what ASLDRO v2.2.0 can
express, so that the two can be compared voxel by voxel. It refuses anything ASLDRO cannot
model. `tools/compat_asldro.py` runs the comparison benchmarks against ASLDRO itself.

### Phantom

The phantom is a directory of NIfTI images on a common grid, each with a JSON sidecar giving its
`Units`:

| File | Contents | Units |
|---|---|---|
| `perfusion.nii.gz` | cerebral blood flow (CBF), mL/100 g/min | `ml/100g/min` |
| `att.nii.gz` | arterial transit time (ATT) | `s` |
| `T1map.nii.gz` | tissue T1 | `s` |
| `T2map.nii.gz` | tissue T2 | `s` |
| `T2starmap.nii.gz` | tissue T2* | `s` |
| `M0map.nii.gz` | equilibrium magnetization of tissue (M0t) | `arbitrary` |
| `dseg.nii.gz` | tissue labels; `dseg.json` names them in a `LabelMap` | `label indices` |
| `fieldmap.nii.gz` | B0 fieldmap (optional) | `Hz` |
| `abv.nii.gz`, `att_arterial.nii.gz` | arterial blood volume (aBV) and the arterial compartment's ATT (optional, for `[macrovascular]`) | `fraction`, `s` |

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
      sub-01_desc-deltam_gt.nii.gz           noise-free ΔM (control minus labeled), per volume
```

Optional features add files:

| Feature | Adds |
|---|---|
| Motion | `ground-truth/sub-01_desc-motion_gt.tsv` and `_desc-motionEvents_gt.tsv`, plus `_desc-deltamStatic_gt` (the unmoved truth). |
| Vascular extensions | `ground-truth/sub-01_desc-deltamIntravascular_gt`, `_desc-deltamArterial_gt`, `_desc-deltamSuppressed_gt`, `_desc-aBV_gt`, `_desc-attArterial_gt`. |
| Physiology | `ground-truth/sub-01_desc-physio_gt.tsv`, with the factors applied per volume and slice (or shot). |
| Several echo times | One series per echo: `sub-01_echo-<n>_part-{mag,phase}_asl.nii.gz` with its own sidecar, and a separate M0 per echo. The one `aslcontext.tsv` and the ground truth are shared by all echoes. |
| Hadamard | The main series is the *decoded* data. `sourcedata/sub-01/perf/` holds the raw encoded series, a table of the raw volumes, the raw truth, and `desc-preparations_gt.tsv` (the factors applied to each labeling). `TotalAcquiredPairs` is the number of encoding cycles. |
| Look-Locker | `ground-truth/sub-01_desc-deltamRead_gt` (what each readout read, after depletion) and `desc-lookLocker_gt.tsv` (the tissue magnetization before each readout). With the vascular extensions, the parts of that read: `_desc-deltamReadIV_gt`, `_desc-deltamReadEV_gt` and `_desc-arterialRead_gt`. |
| 3D gradient-echo EPI | No extra files. The sidecar's `AslscanSimulation.Readout` records the train, the slab entry, the interpolation nodes and the memory estimate. Physiology is recorded per shot. |

The dataset passes the BIDS validator. The `ground-truth/` directories are listed in
`.bidsignore`, and validators skip `sourcedata/` by design. Each `*_asl.json` sidecar repeats the
input protocol and adds an `AslscanSimulation` section. That section records every value aslscan
resolved: grid sizes, kinetic constants and their sources, acquisition settings, random seeds, and
the details of each optional feature. When a setting in the simulation overrides a value in the
input sidecar, the standard BIDS field reports the value that was simulated, and the original
input value is kept under `AslscanSimulation.InputValuesReplaced`.

## Testing

```bash
cargo test --features io,test-hooks
```

```bash
cargo test --release --features cli,kspace,par,test-hooks
```

These run the unit tests and the end-to-end tests, using the fixtures under `tests/fixtures/`.
The second also covers the spiral readout. The reference values for the kinetic and signal models
were generated with ASLDRO by `tools/gen_gkm_fixtures.py` and `tools/gen_mrsignal_fixtures.py`.
The `test-hooks` feature exists only for these tests. Do not enable it when producing data.

Two longer checks need locally converted phantoms under `work/`:

- `tools/regress_identity.sh <aslscan-rev> <mrsim-acq-rev>` builds a base revision and the current
  sources side by side. It compares their outputs byte for byte on a set of protocols covering
  every earlier feature. Use it to show that a change leaves existing outputs unchanged.
- `tools/compat_asldro.py` (run in a Python environment with ASLDRO) compares `--compat-asldro`
  output with ASLDRO voxel by voxel.
- `tools/acceptance_p7.sh <out> [--large]` runs the P7 acceptance series through the BIDS validator.
  With `--large` it also runs the 3D trains on the full 3T phantom and reports run time and memory.

## License

MIT or Apache-2.0, at your option.
