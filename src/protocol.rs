//! BIDS ASL sidecar + `_aslcontext.tsv` + optional TOML overlay -> [`Protocol`], and
//! [`Protocol::acquisition`] -> `mrsim_acq::kspace::Acquisition`.
//!
//! Units: everything in a `Protocol` is in seconds, as BIDS and simasl are, except the two fields
//! whose names say milliseconds. The conversion to the acquisition stage's milliseconds happens
//! in exactly one place, [`Protocol::acquisition`].
//!
//! Precedence for values BIDS does not carry: overlay > sidecar (`LabelingEfficiency` only) >
//! phantom (`phantom.json`) > documented default, and every resolved value remembers its
//! [`Source`] so the output sidecar can record which won. Silent defaults are the failure mode
//! for datasets in the wild.

use std::collections::BTreeMap;
use std::path::Path;

use mrsim_acq::kspace::{Acquisition, EchoFormation, KspaceWindow, PartialFourierMode};
use mrsim_acq::motion::{load_motion_tsv, MotionMode};
use mrsim_acq::readout::{
    centre_echo, check_ge3d_timing, check_grase_timing, check_spiral_timing, esp_from_echo_time, ge3d_lines, grase_block,
    grase_lines, spiral_esp_from_echo_time, spiral_lines, EchoTrain, ExcitationTrain, Ge3dReadout, KzOrder, Readout3d,
};
use serde::Deserialize;
use serde_json::Value;

use crate::kinetic::{Kinetic, LabelType};
use crate::longitudinal::Suppression;
use crate::bolus::Region;
use crate::mrsignal::{parse_contrast, Contrast, IrParams};
use crate::physio::PhysioParams;
use crate::resample::GridOrigin;
pub use crate::rows::{Row, RowKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum M0Type {
    Included,
    Separate,
    Estimate,
    Absent,
}

/// Where a resolved value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Sidecar,
    Phantom,
    Overlay,
    Default,
    /// Inherited from the resolved blood T2 (P4: the arterial T2's default).
    T2Blood,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Sidecar => "Sidecar",
            Source::Phantom => "Phantom",
            Source::Overlay => "Overlay",
            Source::Default => "Default",
            Source::T2Blood => "T2Blood",
        }
    }
}

/// The `phantom.json` block a converted phantom carries; the kinetic constants the oracle used.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhantomParams {
    pub lambda: Option<f64>,
    pub t1b: Option<f64>,
    pub field_strength: Option<f64>,
    /// The phantom carries `abv.nii.gz` / `aatt.nii.gz` (P4, part B), so activation can be
    /// decided here, before the compat check.
    pub has_abv: bool,
    pub has_aatt: bool,
}

/// The acquisition knobs BIDS does not express, all from the overlay with these defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayAcq {
    pub oversample: usize,
    pub matrix: Option<[usize; 2]>,
    pub acs_lines: usize,
    pub ghost_offset: f64,
    pub n_spikes: usize,
    pub spike_amplitude: f64,
    pub n_coils: usize,
    /// ms; the `Acquisition::t_inhom` fallback the per-compartment T2' overrides.
    pub t_inhom_ms: f64,
    pub window: KspaceWindow,
    pub partial_fourier: f64,
    pub pf_mode: PartialFourierMode,
    pub eddy_strength: f64,
    pub eddy_quad: f64,
    pub eddy_phase: f64,
    /// ms.
    pub eddy_tau_ms: f64,
    pub noise_variance: f64,
    pub signal_scale: f64,
}

impl Default for OverlayAcq {
    fn default() -> Self {
        OverlayAcq {
            oversample: 2,
            matrix: None,
            acs_lines: 24,
            ghost_offset: 0.0,
            n_spikes: 0,
            spike_amplitude: 1.0,
            n_coils: 1,
            t_inhom_ms: 50.0,
            window: KspaceWindow::None,
            partial_fourier: 1.0,
            pf_mode: PartialFourierMode::FiberfoxCompatible,
            eddy_strength: 0.0,
            eddy_quad: 0.0,
            eddy_phase: 0.0,
            eddy_tau_ms: 70.0,
            noise_variance: 0.0,
            signal_scale: 100.0,
        }
    }
}

/// The TOML overlay. Every key optional.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Overlay {
    pub seed: Option<u64>,
    pub kinetic: Option<KineticOverlay>,
    pub signal: Option<SignalOverlay>,
    pub acquisition: Option<AcqOverlay>,
    pub m0: Option<M0Overlay>,
    pub background_suppression: Option<SuppressionOverlay>,
    pub motion: Option<MotionOverlay>,
    pub compat: Option<CompatOverlay>,
    pub macrovascular: Option<MacroOverlay>,
    pub vascular_crushing: Option<CrushOverlay>,
    pub physio: Option<PhysioOverlay>,
    pub readout: Option<ReadoutOverlay>,
    pub multi_te: Option<MultiTeOverlay>,
    pub hadamard: Option<HadamardOverlay>,
    pub look_locker: Option<LookLockerOverlay>,
    pub images: Option<ImagesOverlay>,
}

/// `[images]` (P7 addendum, part C): the bound on the compartment images in memory at once.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImagesOverlay {
    /// GiB; default 4. `[multi_te] max_image_memory_gib` is its older name (compat multi-TE).
    pub max_memory_gib: Option<f64>,
}

/// `[look_locker]` (P6 addendum, part B): read only with `LookLocker: true`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookLockerOverlay {
    /// The readouts per cycle, checked against the grouping the arrays give.
    pub readouts_per_cycle: Option<usize>,
}

/// `[hadamard]` (P6 addendum, part A): its presence turns time encoding on.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HadamardOverlay {
    /// The Sylvester order `H` (4, 8, 16 or 32), required: `H - 1` sub-boli per cycle.
    pub order: Option<usize>,
    /// Decode the tissue-only images too and report the leakage (default true).
    pub report_leakage: Option<bool>,
}

/// `[multi_te]` (P6 addendum, part C): read only with more than one echo.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiTeOverlay {
    /// ms reserved about the centre of each spin-echo refocusing pulse (default 2); refused under
    /// gradient echo, which has none.
    pub refocusing_time: Option<f64>,
    /// GiB: the limit on compat's per-echo input images (default 4); compat only.
    pub max_image_memory_gib: Option<f64>,
}

/// `[readout]` (P5 addendum, parts B and C): the 3D echo train and its in-plane readout. Every
/// key is refused with `MRAcquisitionType: 2D`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadoutOverlay {
    /// `"grase"` or `"spiral"`; overrides `PulseSequenceType`.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub ky_segments: Option<usize>,
    pub kz_segments: Option<usize>,
    /// `"centric"` or `"linear"`.
    pub kz_order: Option<String>,
    /// Echo spacing (ms); otherwise from `EchoTime`.
    pub echo_spacing: Option<f64>,
    /// The refocusing pulse and its crushers (ms), centred midway between echoes.
    pub refocusing_time: Option<f64>,
    /// Degrees in `(0, 180]`; otherwise the sidecar's `FlipAngle`, else 180.
    pub refocusing_flip_angle: Option<f64>,
    /// The actual EPI line spacing (ms), GRASE.
    pub line_spacing: Option<f64>,
    /// Receiver samples per line, for the `DwellTime` fallback, GRASE.
    pub readout_samples: Option<usize>,
    /// `"j"` or `"j-"`, GRASE; over the sidecar's.
    pub phase_encoding_direction: Option<String>,
    /// Spiral interleaves (shots per kz segment), required for a spiral.
    pub interleaves: Option<usize>,
    /// Spiral readout duration (ms), required for a spiral.
    pub spiral_readout_time: Option<f64>,
    /// Spiral dwell time (s); otherwise the sidecar's `DwellTime`.
    pub dwell_time: Option<f64>,
    /// How much denser than the Nyquist spacing a spiral's turns are (at least 1; default
    /// [`DEFAULT_RADIAL_OVERSAMPLING`]).
    pub radial_oversampling: Option<f64>,
    /// `"epi3d"` (P7 part C): the excitation spacing within a train (ms), required.
    pub excitation_spacing: Option<f64>,
    /// `"epi3d"`: the excitation pulse reserved about each excitation (ms, default 2).
    pub excitation_time: Option<f64>,
    /// `"epi3d"`: seconds from labeling to the label's entry into the slab, or `"arrival"` (P7
    /// part C, depletion from slab entry); otherwise bolus-position's.
    pub slab_entry_time: Option<SlabEntry>,
    /// `"epi3d"`: the tissue compartments' distinct-T1 groups allowed (default 16).
    pub max_t1_groups: Option<usize>,
    /// `"epi3d"`: the blood interpolation's tolerance, relative to its peak (default 1e-4).
    pub node_tolerance: Option<f64>,
}

/// `[macrovascular]` (P4, part B): per-label values keyed by `dseg.json` names.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MacroOverlay {
    pub arterial_blood_volume: Option<BTreeMap<String, f64>>,
    pub arterial_transit_time: Option<BTreeMap<String, f64>>,
}

/// `[vascular_crushing]` (P4, part C).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrushOverlay {
    /// cm/s per label: the top of the laminar speed range.
    pub arterial_velocity: Option<BTreeMap<String, f64>>,
    /// Accept `VascularCrushing: true` with no arterial compartment to act on.
    pub no_arterial_compartment: Option<bool>,
}

/// `[physio]` (P4, part E).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysioOverlay {
    pub tissue_cardiac: Option<f64>,
    pub tissue_respiratory: Option<f64>,
    pub tissue_drift: Option<f64>,
    pub label_cardiac: Option<f64>,
    pub label_respiratory: Option<f64>,
    pub label_drift: Option<f64>,
    pub cardiac_frequency: Option<f64>,
    pub cardiac_cv: Option<f64>,
    pub respiratory_frequency: Option<f64>,
    pub respiratory_cv: Option<f64>,
    /// s
    pub drift_time: Option<f64>,
}

/// `[background_suppression] slab_entry_time`: seconds, or `"arrival"`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SlabEntry {
    Seconds(f64),
    Word(String),
}

/// Where an arterial quantity comes from (P4, part B).
#[derive(Debug, Clone, PartialEq)]
pub enum QuantitySource {
    Map,
    Table(BTreeMap<String, f64>),
}

/// The resolved arterial compartment.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroSpec {
    pub abv: QuantitySource,
    pub aatt: QuantitySource,
    /// s
    pub t2_arterial: (f64, Source),
}

/// The resolved vascular crushing.
#[derive(Debug, Clone, PartialEq)]
pub struct CrushSpec {
    /// cm/s per row; 0 is crushing off for that volume.
    pub venc: Vec<f64>,
    /// cm/s per label; `None` with `no_arterial_compartment`.
    pub arterial_velocity: Option<BTreeMap<String, f64>>,
    pub no_arterial_compartment: bool,
}

/// The suppression model (P4, part D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SuppressionModel {
    /// P3: every pulse inverts the whole bolus.
    GlobalBolus,
    BolusPosition(Region),
}

/// `[compat]` (P2 addendum, part A).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatOverlay {
    /// Pin the acquisition to what simasl (ASLDRO v2.2.0) can express; the CLI's
    /// `--compat-asldro` sets it.
    pub asldro: Option<bool>,
    /// simasl's SNR, converted to the acquisition's noise variance; 0 or absent is no noise.
    /// Read only with `asldro = true`.
    pub desired_snr: Option<f64>,
    /// `"corner"` (P1) or `"voxel-centre"` (simasl); default `voxel-centre` under `asldro`,
    /// `corner` otherwise.
    pub grid_origin: Option<String>,
}

/// The resolved compat mode (`asldro = true`).
#[derive(Debug, Clone, PartialEq)]
pub struct CompatSpec {
    /// `None` when absent or 0: no noise.
    pub desired_snr: Option<f64>,
}

/// The largest accepted `[vascular_crushing] arterial_velocity` (cm/s).
pub const MAX_ARTERIAL_VELOCITY: f64 = 1000.0;
/// The smallest accepted `[kinetic] exchange_time` (s).
pub const MIN_EXCHANGE_TIME: f64 = 1e-6;
/// The accepted range of `[physio]` cardiac and respiratory frequencies (Hz): periods of 0.1 to
/// 100 s, so a period is finite and a series holds a bounded number of them.
pub const PHYSIO_FREQUENCY_RANGE: (f64, f64) = (0.01, 10.0);

/// A spiral's default radial oversampling (P5 part C, amended): its turns 1/1.2 cycle/FOV apart,
/// the margin at which the reconstruction recovers every in-band image direction.
pub const DEFAULT_RADIAL_OVERSAMPLING: f64 = 1.2;

/// The acquisition values `asldro = true` pins, as `(overlay key, value)`: what simasl's
/// acquisition can express (no readout effects, one coil, full sampling, unit scale, and the
/// noise coming from `desired_snr` instead of `noise_variance`).
pub const COMPAT_PINNED: [(&str, f64); 11] = [
    ("oversample", 1.0),
    ("partial_fourier", 1.0),
    ("n_coils", 1.0),
    ("ghost_offset", 0.0),
    ("n_spikes", 0.0),
    ("eddy_strength", 0.0),
    ("eddy_quad", 0.0),
    ("eddy_phase", 0.0),
    ("signal_scale", 1.0),
    ("noise_variance", 0.0),
    ("ParallelReductionFactorInPlane", 1.0),
];

/// `[background_suppression]` (P3 addendum, part A). Read only when the sidecar's
/// `BackgroundSuppression` is true.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuppressionOverlay {
    /// The fraction of longitudinal magnetization each pulse inverts; default 0.95.
    pub inversion_efficiency: Option<f64>,
    /// A saturation pulse on the imaging region at labeling start; default false.
    pub presaturation: Option<bool>,
    /// Multi-PLD series: one pulse-time array per distinct PostLabelingDelay, ascending.
    pub pulse_times_per_pld: Option<Vec<Vec<f64>>>,
    /// `"global-bolus"` (P3, default) or `"bolus-position"` (P4, part D).
    pub model: Option<String>,
    /// `"global"` or `"slab"`; required by and only read with `"bolus-position"`.
    pub pulse_region: Option<String>,
    /// With `"slab"`: seconds from labeling to slab entry, or `"arrival"`.
    pub slab_entry_time: Option<SlabEntry>,
}

/// `[motion]` (P3 addendum, part C).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionOverlay {
    /// `off` (default) | `trajectory` | `random` | `linear`.
    pub mode: Option<String>,
    /// TSV path (confounds format: `trans_x/y/z` mm, `rot_x/y/z` **radians**), one row per
    /// volume; relative to the overlay file when read through [`load`].
    pub trajectory: Option<String>,
    /// Per-axis amplitudes (mm) for `random` and `linear`.
    pub trans_mm: Option<[f64; 3]>,
    /// Per-axis amplitudes (degrees) for `random` and `linear`.
    pub rot_deg: Option<[f64; 3]>,
    /// The volumes `random`/`linear` affect; default all.
    pub volumes: Option<Vec<usize>>,
    pub within_volume: Option<WithinVolumeOverlay>,
}

/// `[motion.within_volume]`: multiband shot events (needs `MultibandAccelerationFactor > 1`).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WithinVolumeOverlay {
    /// Probability that a shot has an event, in `[0, 1]`.
    pub dropout_rate: f64,
    /// Signal attenuation of the event shot's slices, in `[0, 1]` (`DropoutLaw::Uniform`).
    pub severity: f64,
    /// Per-axis jump amplitudes (mm, degrees) drawn in `[-amp, amp]`, persisting for the rest of
    /// the volume; default zero (pure dropout).
    pub jump_mm: Option<[f64; 3]>,
    pub jump_deg: Option<[f64; 3]>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KineticOverlay {
    pub label_efficiency: Option<f64>,
    pub lambda_blood_brain: Option<f64>,
    pub t1_arterial_blood: Option<f64>,
    /// s; turns the intravascular/extravascular split on (P4, part A).
    pub exchange_time: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalOverlay {
    pub acq_contrast: Option<String>,
    /// s
    pub t2_blood: Option<f64>,
    /// s; the arterial compartment's T2 (P4, part B), default the resolved blood T2.
    pub t2_arterial: Option<f64>,
    /// s; inversion recovery only (overlay > sidecar `InversionTime` > 1.0).
    pub inversion_time: Option<f64>,
    /// degrees; inversion recovery only (overlay > sidecar `FlipAngle` > 90). Undefined for
    /// spin echo, whose equation assumes 90.
    pub excitation_flip_angle: Option<f64>,
    /// degrees; inversion recovery only (default 180).
    pub inversion_flip_angle: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcqOverlay {
    pub oversample: Option<usize>,
    pub matrix: Option<[usize; 2]>,
    pub acs_lines: Option<usize>,
    pub ghost_offset: Option<f64>,
    pub n_spikes: Option<usize>,
    pub spike_amplitude: Option<f64>,
    pub n_coils: Option<usize>,
    /// ms
    pub t_inhom: Option<f64>,
    /// `none` | `hann` | `tukey:<alpha>` | `fermi:<radius>,<width>`
    pub window: Option<String>,
    pub partial_fourier: Option<f64>,
    /// `fiberfox` | `contiguous`
    pub pf_mode: Option<String>,
    pub eddy_strength: Option<f64>,
    pub eddy_quad: Option<f64>,
    pub eddy_phase: Option<f64>,
    /// ms
    pub eddy_tau: Option<f64>,
    pub noise_variance: Option<f64>,
    pub signal_scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M0Overlay {
    /// s; the separate M0 scan's repetition time, which nothing in the ASL sidecar carries.
    pub repetition_time: Option<f64>,
    /// Degrees: a Look-Locker series' separate M0 excitation (P6 part B); required when the
    /// series' FlipAngle is an array.
    pub flip_angle: Option<f64>,
}

/// The resolved background suppression: the pulse set per row (empty for m0scan rows), the
/// efficiency and the presaturation flag with their sources.
#[derive(Debug, Clone, PartialEq)]
pub struct SuppressionSpec {
    pub epsilon: (f64, Source),
    pub presaturation: (bool, Source),
    /// Seconds from labeling start, per row in series order.
    pub per_row: Vec<Vec<f64>>,
    /// The first (lowest) PLD's pulse times: what BIDS' `BackgroundSuppressionPulseTime`
    /// carries, so the effective standard field is written from here.
    pub first_pld_pulses: Vec<f64>,
    /// A multi-PLD series without `pulse_times_per_pld`: BIDS defines only the first PLD's
    /// times, and they were applied to every row.
    pub first_pld_applied_to_all: bool,
    /// P3's global bolus, or P4's bolus position.
    pub model: SuppressionModel,
}

impl SuppressionSpec {
    pub fn for_row(&self, i: usize) -> Suppression {
        Suppression::new(self.per_row[i].clone(), self.epsilon.0, self.presaturation.0)
    }
}

/// The resolved inversion-recovery parameters with their sources.
#[derive(Debug, Clone, PartialEq)]
pub struct IrSpec {
    pub params: IrParams,
    pub inversion_time: Source,
    pub excitation_flip: Source,
    pub inversion_flip: Source,
}

/// The 3D in-plane readout (P5 addendum, parts B and C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadoutKind {
    Grase,
    Spiral,
    /// The 3D gradient-echo stack-of-EPI train (P7 addendum, part C).
    Epi3d,
}

impl ReadoutKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReadoutKind::Grase => "grase",
            ReadoutKind::Spiral => "spiral",
            ReadoutKind::Epi3d => "epi3d",
        }
    }
}

/// A 3D protocol's readout as parsed: everything that does not need the acquisition grid (P5
/// addendum, part B, "Inputs"; plan, Task 7). [`resolve_readout`] completes it with the grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutSpec {
    pub kind: (ReadoutKind, Source),
    /// `NumberShots`, 1 when absent (GRASE).
    pub number_shots: (usize, Source),
    pub ky_segments: (usize, Source),
    pub kz_segments: (usize, Source),
    pub kz_order: (KzOrder, Source),
    pub echo_spacing_ms: Option<f64>,
    pub refocusing_time_ms: (f64, Source),
    /// The refocusing angle: the sidecar's `FlipAngle` read as such (the excitation is 90).
    pub refocusing_flip_deg: (f64, Source),
    pub line_spacing_ms: Option<f64>,
    pub readout_samples: Option<usize>,
    pub effective_echo_spacing_s: Option<f64>,
    pub total_readout_time_s: Option<f64>,
    /// The dwell time (s): GRASE's line-spacing fallback (sidecar only); a spiral's sampling
    /// interval (overlay over sidecar).
    pub dwell_time_s: Option<(f64, Source)>,
    /// Spirals (part C): interleaves and the readout duration (ms), both required.
    pub interleaves: Option<(usize, Source)>,
    pub spiral_readout_ms: Option<(f64, Source)>,
    pub radial_oversampling: Option<(f64, Source)>,
    /// P7 part C, `"epi3d"` only: the excitation spacing and pulse (ms), the slab entry (from
    /// `[readout]` or bolus-position's), the tissue groups allowed and the interpolation tolerance.
    pub ge3d: Option<Ge3dSpec>,
}

/// Where the label enters the slab (P7 addendum, part C): `d` seconds after its labeling, or at its
/// arrival in the voxel (no depletion before it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlabEntryTime {
    Seconds(f64),
    Arrival,
}

/// The `"epi3d"` readout's own inputs (P7 addendum, part C, "Inputs").
#[derive(Debug, Clone, PartialEq)]
pub struct Ge3dSpec {
    pub excitation_spacing_ms: f64,
    pub excitation_time_ms: (f64, Source),
    /// The slab entry and where it came from (`[readout]` or `[background_suppression]`).
    pub slab_entry: (SlabEntryTime, &'static str),
    pub max_t1_groups: (usize, Source),
    pub node_tolerance: (f64, Source),
    /// The bound on the images in memory at once (GiB): `[images] max_memory_gib`, default 4.
    pub max_memory_gib: (f64, Source),
}

/// The readout completed with the acquisition grid (P5 plan, Task 7): the echo train, the line
/// table's inputs, and every value the sidecar records.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutResolution {
    pub train: EchoTrain,
    pub readout: Readout3d,
    pub n_shots: usize,
    /// Lines per EPI block and echoes per train.
    pub epi: usize,
    pub etl: usize,
    pub t_line_ms: f64,
    pub t_line_source: &'static str,
    /// BIDS's effective spacing (s), when one is defined.
    pub effective_spacing_s: Option<f64>,
    pub esp_ms: f64,
    /// The echo reading the kz centre, and the centre line's time from its echo (ms; zero for a
    /// spiral, which starts at its echo).
    pub e_c: usize,
    pub t_kyc_ms: f64,
    /// A spiral's in-plane design (part C); `None` for GRASE. For a spiral the GRASE-only fields
    /// above (`epi`, `t_line_ms`, `t_line_source`, `effective_spacing_s`) are zero or empty.
    pub spiral: Option<SpiralResolution>,
}

/// A spiral readout as resolved (P5 addendum, part C, "Trajectory" and "Inputs").
#[derive(Debug, Clone, PartialEq)]
pub struct SpiralResolution {
    pub interleaves: usize,
    pub readout_ms: f64,
    pub dwell_ms: f64,
    pub dwell_source: Source,
    pub samples_per_interleaf: usize,
    pub k_max: f64,
    pub n_turns: f64,
    /// The constant-angular-velocity centre region the sampling bound requires (ms).
    pub tau_c_ms: f64,
    /// The turns' density over the Nyquist spacing, and its source.
    pub radial_oversampling: (f64, Source),
}

/// The resolved gradient-echo excitation (P5 addendum, part A) with its source.
#[derive(Debug, Clone, PartialEq)]
pub struct GeSpec {
    /// Signed degrees in `[-180, 180]` (BIDS's `FlipAngle` folded as IR's is).
    pub flip_deg: f64,
    pub flip: Source,
}

/// Within-volume (multiband shot) motion events.
#[derive(Debug, Clone, PartialEq)]
pub struct WithinVolume {
    pub dropout_rate: f64,
    pub severity: f64,
    pub jump_mm: [f64; 3],
    pub jump_deg: [f64; 3],
}

/// The resolved motion request.
#[derive(Debug, Clone)]
pub struct MotionSpec {
    pub mode: MotionMode,
    /// The overlay's mode name, for the sidecar.
    pub mode_name: String,
    pub within: Option<WithinVolume>,
}

/// One Hadamard encoding cycle (P6 addendum, part A): its `deltam` rows, in sub-bolus order.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardCycle {
    pub rows: Vec<usize>,
}

/// Hadamard time-encoded labeling (P6 addendum, part A), resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardSpec {
    pub order: usize,
    pub report_leakage: (bool, Source),
    /// Each sub-bolus's duration (s), in labeling order (sub-bolus 1 is labeled first).
    pub tau: Vec<f64>,
    /// Each sub-bolus's span `[a_j, b_j]` within the labeling `[0, tau_tot]` (s).
    pub spans: Vec<(f64, f64)>,
    pub tau_tot: f64,
    /// The delay from the end of the labeling to the excitation (s): the last sub-bolus's PLD
    /// (under Look-Locker, the first readout's).
    pub pld: f64,
    pub cycles: Vec<HadamardCycle>,
    /// P7 part B: the Look-Locker readouts after each encoded preparation (1 without Look-Locker)
    /// and each readout's delay from the end of the labeling, `PLD_n` (`[pld]` without).
    pub readouts: usize,
    pub plds: Vec<f64>,
}

impl HadamardSpec {
    /// The input row of decoded volume (sub-bolus `j`, readout `n`) within a cycle's rows: the
    /// rows are readout-major, `H - 1` sub-boli per readout.
    pub fn row(&self, cy: &HadamardCycle, j: usize, n: usize) -> usize {
        cy.rows[n * (self.order - 1) + j]
    }
}

/// One Look-Locker cycle (P6 addendum, part B): its rows (the readouts in order, or one m0scan row).
#[derive(Debug, Clone, PartialEq)]
pub struct LookLockerCycle {
    pub rows: Vec<usize>,
    pub m0scan: bool,
}

/// Look-Locker readouts (P6 addendum, part B), resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct LookLockerSpec {
    pub cycles: Vec<LookLockerCycle>,
    /// The excitation flip per row (degrees, in (0, 90]); an m0scan row's is its own.
    pub flip_deg: Vec<f64>,
    /// Whether `FlipAngle` was an array.
    pub flip_array: bool,
    pub readouts_per_cycle: Option<usize>,
    /// The separate M0's excitation (degrees) and where it came from.
    pub m0_flip_deg: Option<(f64, Source)>,
}

/// Multi-TE (P6 addendum, part C): one input sidecar per echo.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTeSpec {
    /// The input sidecar of each echo, in echo order (the first is also `Protocol::input_sidecar`).
    pub sidecars: Vec<Value>,
    /// ms reserved about each refocusing pulse; `None` under gradient echo.
    pub refocusing_time_ms: Option<(f64, Source)>,
    /// The limit on compat's per-echo input images (GiB); `None` outside compat.
    pub max_image_memory_gib: Option<(f64, Source)>,
}

/// The resolved protocol. Seconds unless the field name says `_ms`.
#[derive(Debug, Clone)]
pub struct Protocol {
    pub label_type: LabelType,
    pub rows: Vec<Row>,
    pub m0_type: M0Type,
    pub background_suppression: bool,
    /// `Some` when `background_suppression` is true (possibly with zero pulses).
    pub suppression: Option<SuppressionSpec>,
    /// `Some` when `contrast` is inversion recovery.
    pub ir: Option<IrSpec>,
    /// `Some` when `contrast` is gradient echo.
    pub ge: Option<GeSpec>,
    /// `Some` for `MRAcquisitionType: 3D` (P5 part B): the readout as parsed; [`resolve_readout`]
    /// completes it once the acquisition grid is known.
    pub readout: Option<ReadoutSpec>,
    /// `Some` when the overlay asks for motion (a mode other than `off`, or shot events).
    pub motion: Option<MotionSpec>,
    /// `Some` under `[compat] asldro = true` (P2).
    pub compat: Option<CompatSpec>,
    /// Where the acquisition grid sits on the phantom (`[compat] grid_origin`).
    pub grid_origin: GridOrigin,
    /// P4, part A: the exchange time (s).
    pub exchange_time: Option<f64>,
    /// P4, part B.
    pub macrovascular: Option<MacroSpec>,
    /// P4, part C (`VascularCrushing: true`).
    pub crushing: Option<CrushSpec>,
    /// P4, part E.
    pub physio: Option<PhysioParams>,
    /// Each row's start on the series clock (s): the sum of the earlier rows' repetition times.
    pub row_start: Vec<f64>,
    /// With `mb > 1`: whether the slice timing is the interleaved (even groups then odd) shot
    /// order of `mrsim_acq::motion::slice_schedule`, as opposed to sequential.
    pub mb_interleaved: bool,
    /// Per acquired slice in data z order (after `SliceEncodingDirection`), minus the minimum (s).
    pub slice_offsets: Vec<f64>,
    pub field_strength: f64,
    pub voxel_size_mm: [f64; 3],
    pub reverse_phase: bool,
    pub phase_encoding_direction: String,
    pub echo_time_s: f64,
    /// Every echo's time (s), increasing; one entry for a single echo (`echo_time_s`).
    pub echo_times_s: Vec<f64>,
    /// `Some` with more than one echo.
    pub multi_te: Option<MultiTeSpec>,
    /// `Some` under `[hadamard]` (P6 part A); `rows` are then the decoded rows.
    pub hadamard: Option<HadamardSpec>,
    /// `Some` with `LookLocker: true` (P6 part B); `rows` are then the readout volumes.
    pub look_locker: Option<LookLockerSpec>,
    pub total_readout_time_s: f64,
    pub accel: usize,
    pub mb: usize,
    pub alpha: (f64, Source),
    pub lambda: (f64, Source),
    pub t1b: (f64, Source),
    pub t2_blood_s: (f64, Source),
    pub contrast: Contrast,
    pub m0_repetition_time_s: Option<f64>,
    pub seed: u64,
    pub acq: OverlayAcq,
    /// The input sidecar, echoed verbatim into the output sidecar.
    pub input_sidecar: Value,
}

// ---------------------------------------------------------------- JSON helpers

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, String> {
    v.get(key).ok_or_else(|| format!("asl.json: missing required field {key:?}"))
}

fn num(v: &Value, key: &str) -> Result<f64, String> {
    field(v, key)?.as_f64().ok_or_else(|| format!("asl.json: {key} must be a number"))
}

fn opt_num(v: &Value, key: &str) -> Result<Option<f64>, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(x) => x.as_f64().map(Some).ok_or_else(|| format!("asl.json: {key} must be a number")),
    }
}

fn string(v: &Value, key: &str) -> Result<String, String> {
    field(v, key)?.as_str().map(str::to_string).ok_or_else(|| format!("asl.json: {key} must be a string"))
}

fn opt_bool(v: &Value, key: &str) -> Result<bool, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("asl.json: {key} must be a boolean")),
    }
}

/// A number, or a non-empty array of finite numbers. The flag says which it was: a one-element
/// array is still an array and is held to the per-volume length rule.
fn num_or_array(v: &Value, key: &str) -> Result<(Vec<f64>, bool), String> {
    let (vals, is_array) = match field(v, key)? {
        Value::Number(n) => (vec![n.as_f64().unwrap()], false),
        Value::Array(a) => (
            a.iter()
                .map(|x| x.as_f64().ok_or_else(|| format!("asl.json: {key} array holds a non-number")))
                .collect::<Result<Vec<f64>, String>>()?,
            true,
        ),
        _ => return Err(format!("asl.json: {key} must be a number or an array of numbers")),
    };
    if vals.is_empty() {
        return Err(format!("asl.json: {key} is an empty array"));
    }
    if let Some(bad) = vals.iter().find(|x| !x.is_finite()) {
        return Err(format!("asl.json: {key} holds a non-finite value {bad}"));
    }
    Ok((vals, is_array))
}

fn require_finite_positive(v: f64, what: &str) -> Result<f64, String> {
    if v.is_finite() && v > 0.0 { Ok(v) } else { Err(format!("{what} must be a positive finite number, got {v}")) }
}

fn require_finite_nonneg(v: f64, what: &str) -> Result<f64, String> {
    if v.is_finite() && v >= 0.0 { Ok(v) } else { Err(format!("{what} must be a non-negative finite number, got {v}")) }
}

/// Expand a scalar-or-array timing field to one value per row. BIDS: the array form has one
/// entry per volume in acquisition order, m0scan rows included and set to zero.
fn per_row((vals, is_array): (Vec<f64>, bool), key: &str, rows: &[RowKind], zero_for_m0: bool)
    -> Result<Vec<f64>, String>
{
    if !is_array {
        return Ok(rows.iter().map(|k| if zero_for_m0 && *k == RowKind::M0scan { 0.0 } else { vals[0] }).collect());
    }
    if vals.len() != rows.len() {
        return Err(format!(
            "asl.json: {key} has {} entries but aslcontext.tsv has {} rows; BIDS requires one entry \
             per volume, m0scan rows included",
            vals.len(), rows.len()));
    }
    if zero_for_m0 {
        for (i, (v, k)) in vals.iter().zip(rows).enumerate() {
            if *k == RowKind::M0scan && *v != 0.0 {
                return Err(format!("asl.json: {key}[{i}] = {v} on an m0scan row; BIDS requires 0"));
            }
        }
    }
    Ok(vals)
}

// ---------------------------------------------------------------- aslcontext.tsv

/// Parse `_aslcontext.tsv`: a `volume_type` header, then one kind per line. Blank lines are
/// tolerated (bids-examples asl004 ends with one).
pub fn parse_aslcontext(text: &str) -> Result<Vec<RowKind>, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    match lines.next() {
        Some("volume_type") => {}
        other => return Err(format!("aslcontext.tsv: expected a `volume_type` header, got {other:?}")),
    }
    let mut rows = Vec::new();
    for (i, l) in lines.enumerate() {
        rows.push(match l {
            "m0scan" => RowKind::M0scan,
            "control" => RowKind::Control,
            "label" => RowKind::Label,
            "deltam" => RowKind::Deltam,
            "cbf" => {
                return Err(format!(
                    "aslcontext.tsv row {i}: `cbf` is a quantified output, not an acquired volume; \
                     P1 does not simulate it"))
            }
            other => return Err(format!("aslcontext.tsv row {i}: unknown volume_type {other:?}")),
        });
    }
    if rows.is_empty() {
        return Err("aslcontext.tsv has no volumes".to_string());
    }
    Ok(rows)
}

// ---------------------------------------------------------------- overlay pieces

fn parse_window(s: &str) -> Result<KspaceWindow, String> {
    let (name, arg) = s.split_once(':').unwrap_or((s, ""));
    match name.to_ascii_lowercase().as_str() {
        "none" => Ok(KspaceWindow::None),
        "hann" => Ok(KspaceWindow::Hann),
        "tukey" => Ok(KspaceWindow::Tukey {
            alpha: arg.parse().map_err(|_| format!("window {s:?}: expected tukey:<alpha>"))?,
        }),
        "fermi" => {
            let (r, w) = arg.split_once(',').ok_or_else(|| format!("window {s:?}: expected fermi:<radius>,<width>"))?;
            Ok(KspaceWindow::Fermi {
                radius: r.trim().parse().map_err(|_| format!("window {s:?}: bad radius"))?,
                width: w.trim().parse().map_err(|_| format!("window {s:?}: bad width"))?,
            })
        }
        _ => Err(format!("window {s:?}: expected none | hann | tukey:<alpha> | fermi:<radius>,<width>")),
    }
}

fn parse_pf_mode(s: &str) -> Result<PartialFourierMode, String> {
    match s.to_ascii_lowercase().as_str() {
        "fiberfox" => Ok(PartialFourierMode::FiberfoxCompatible),
        "contiguous" => Ok(PartialFourierMode::Contiguous),
        _ => Err(format!("pf_mode {s:?}: expected fiberfox | contiguous")),
    }
}

fn overlay_acq(o: Option<&AcqOverlay>) -> Result<OverlayAcq, String> {
    let mut a = OverlayAcq::default();
    let Some(o) = o else { return Ok(a) };
    if let Some(v) = o.oversample { a.oversample = v; }
    if let Some(v) = o.matrix { a.matrix = Some(v); }
    if let Some(v) = o.acs_lines { a.acs_lines = v; }
    if let Some(v) = o.ghost_offset { a.ghost_offset = v; }
    if let Some(v) = o.n_spikes { a.n_spikes = v; }
    if let Some(v) = o.spike_amplitude { a.spike_amplitude = v; }
    if let Some(v) = o.n_coils { a.n_coils = v; }
    if let Some(v) = o.t_inhom { a.t_inhom_ms = v; }
    if let Some(v) = &o.window { a.window = parse_window(v)?; }
    if let Some(v) = o.partial_fourier { a.partial_fourier = v; }
    if let Some(v) = &o.pf_mode { a.pf_mode = parse_pf_mode(v)?; }
    if let Some(v) = o.eddy_strength { a.eddy_strength = v; }
    if let Some(v) = o.eddy_quad { a.eddy_quad = v; }
    if let Some(v) = o.eddy_phase { a.eddy_phase = v; }
    if let Some(v) = o.eddy_tau { a.eddy_tau_ms = v; }
    if let Some(v) = o.noise_variance { a.noise_variance = v; }
    if let Some(v) = o.signal_scale { a.signal_scale = v; }
    if a.oversample == 0 {
        return Err("overlay: acquisition.oversample must be at least 1".to_string());
    }
    if let Some([nx, ny]) = a.matrix {
        if nx == 0 || ny == 0 {
            return Err("overlay: acquisition.matrix entries must be positive".to_string());
        }
    }
    if a.n_coils == 0 {
        return Err("overlay: acquisition.n_coils must be at least 1".to_string());
    }
    require_finite_positive(a.t_inhom_ms, "overlay acquisition.t_inhom")?;
    require_finite_positive(a.eddy_tau_ms, "overlay acquisition.eddy_tau")?;
    require_finite_positive(a.signal_scale, "overlay acquisition.signal_scale")?;
    require_finite_nonneg(a.noise_variance, "overlay acquisition.noise_variance")?;
    require_finite_nonneg(a.spike_amplitude, "overlay acquisition.spike_amplitude")?;
    if !(a.partial_fourier.is_finite() && a.partial_fourier > 0.0 && a.partial_fourier <= 1.0) {
        return Err(format!("overlay: acquisition.partial_fourier must be in (0, 1], got {}", a.partial_fourier));
    }
    for (name, v) in [("ghost_offset", a.ghost_offset), ("eddy_strength", a.eddy_strength),
                      ("eddy_quad", a.eddy_quad), ("eddy_phase", a.eddy_phase)] {
        if !v.is_finite() {
            return Err(format!("overlay: acquisition.{name} is not finite"));
        }
    }
    Ok(a)
}

/// With `mb > 1`, check that the per-slice offsets (data z order) are a schedule
/// `mrsim_acq::motion::slice_schedule` can represent: the slices sharing each excitation are
/// `{g, g + n_groups, ...}`, and the groups fire sequentially or interleaved (even groups then
/// odd). Returns the `interleaved` flag. P1's "each time occurs `mb` times" check is necessary
/// but not sufficient: `[0, 0, 1, 1]` with `mb = 2` passes it and would be grouped `{0, 2}`,
/// `{1, 3}` by the motion module.
fn multiband_schedule(offsets: &[f64], mb: usize) -> Result<bool, String> {
    let nz = offsets.len();
    if !nz.is_multiple_of(mb) {
        return Err(format!(
            "asl.json: SliceTiming has {nz} slices, not a multiple of MultibandAccelerationFactor {mb}"));
    }
    let n_groups = nz / mb;
    let mut distinct: Vec<u64> = offsets.iter().map(|t| t.to_bits()).collect();
    distinct.sort_unstable();
    distinct.dedup();
    let mut order = Vec::with_capacity(n_groups);
    for b in &distinct {
        let members: Vec<usize> = (0..nz).filter(|&z| offsets[z].to_bits() == *b).collect();
        let g = members[0];
        let want: Vec<usize> = (0..mb).map(|j| g + j * n_groups).collect();
        if g >= n_groups || members != want {
            return Err(format!(
                "asl.json: SliceTiming excites slices {members:?} together, but with {nz} slices and \
                 MultibandAccelerationFactor {mb} a shot excites slices {{g, g + {n_groups}, ...}}; \
                 that is the only slice grouping the motion module's schedule represents"));
        }
        order.push(g);
    }
    let sequential: Vec<usize> = (0..n_groups).collect();
    let interleaved: Vec<usize> = (0..n_groups).step_by(2).chain((1..n_groups).step_by(2)).collect();
    if order == sequential {
        Ok(false)
    } else if order == interleaved {
        Ok(true)
    } else {
        Err(format!(
            "asl.json: SliceTiming fires the shot groups in order {order:?}; sequential {sequential:?} \
             or interleaved {interleaved:?} are the schedules the motion module represents"))
    }
}

/// `mrsim_acq::motion::load_motion_tsv` maps a missing or unparseable cell to 0 and accepts a
/// non-finite number, so a truncated row or a NaN would silently change the trajectory. Check
/// the six columns of every row first; `n/a` is the documented sentinel for 0.
fn check_motion_tsv(path: &str) -> Result<(), String> {
    const COLS: [&str; 6] = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"];
    let text = std::fs::read_to_string(path).map_err(|e| format!("overlay motion.trajectory {path}: {e}"))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().ok_or_else(|| format!("overlay motion.trajectory {path}: empty TSV"))?.split('\t').collect();
    let idx = COLS
        .iter()
        .map(|c| header.iter().position(|h| h.trim() == *c).ok_or_else(|| format!("overlay motion.trajectory {path}: column {c:?} missing")))
        .collect::<Result<Vec<usize>, String>>()?;
    for (i, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split('\t').collect();
        for (&c, name) in idx.iter().zip(COLS) {
            let cell = cells.get(c).map(|s| s.trim())
                .ok_or_else(|| format!("overlay motion.trajectory {path} row {i}: column {name} is missing"))?;
            if cell == "n/a" {
                continue;
            }
            match cell.parse::<f64>() {
                Ok(v) if v.is_finite() => {}
                _ => return Err(format!("overlay motion.trajectory {path} row {i}: {name} = {cell:?} is not a finite number or n/a")),
            }
        }
    }
    Ok(())
}

fn vec3_amplitude(v: Option<[f64; 3]>, what: &str) -> Result<[f64; 3], String> {
    let v = v.unwrap_or([0.0; 3]);
    for x in v {
        require_finite_nonneg(x, what)?;
    }
    Ok(v)
}

/// `shots_3d`: a 3D protocol's NumberShots (P5 part D: there a shot is a readout segment).
fn overlay_motion(m: Option<&MotionOverlay>, n: usize, mb: usize, shots_3d: Option<usize>) -> Result<Option<MotionSpec>, String> {
    let Some(m) = m else { return Ok(None) };
    let mode_name = m.mode.clone().unwrap_or_else(|| "off".to_string()).to_ascii_lowercase();
    let volumes: Vec<usize> = match &m.volumes {
        Some(v) => {
            if let Some(bad) = v.iter().find(|&&i| i >= n) {
                return Err(format!("overlay: motion.volumes index {bad} is outside the {n} volumes"));
            }
            // mrsim-acq's linear mode divides the amplitude by the list length but steps once
            // per distinct volume, and random mode redraws a duplicate: reject repeats.
            let mut seen = vec![false; n];
            for &i in v {
                if std::mem::replace(&mut seen[i], true) {
                    return Err(format!("overlay: motion.volumes lists volume {i} more than once"));
                }
            }
            v.clone()
        }
        None => (0..n).collect(),
    };
    let trans_mm = vec3_amplitude(m.trans_mm, "overlay motion.trans_mm")?;
    let rot_deg = vec3_amplitude(m.rot_deg, "overlay motion.rot_deg")?;
    if mode_name != "trajectory" && m.trajectory.is_some() {
        return Err(format!("overlay: motion.trajectory is only read with mode = \"trajectory\" (mode is {mode_name:?})"));
    }
    let mode = match mode_name.as_str() {
        "off" => MotionMode::Off,
        "random" | "linear" => {
            if trans_mm == [0.0; 3] && rot_deg == [0.0; 3] {
                return Err(format!(
                    "overlay: motion.mode {mode_name:?} with zero trans_mm and rot_deg moves nothing; set an \
                     amplitude or mode = \"off\""));
            }
            if mode_name == "random" {
                MotionMode::Random { trans_mm, rot_deg, volumes }
            } else {
                MotionMode::Linear { trans_mm, rot_deg, volumes }
            }
        }
        "trajectory" => {
            let path = m.trajectory.as_ref()
                .ok_or("overlay: motion.mode \"trajectory\" needs motion.trajectory, a TSV path")?;
            check_motion_tsv(path)?;
            let poses = load_motion_tsv(Path::new(path)).map_err(|e| format!("overlay motion.trajectory {path}: {e}"))?;
            if poses.len() != n {
                return Err(format!(
                    "overlay motion.trajectory {path} has {} rows but the series has {n} volumes", poses.len()));
            }
            MotionMode::Trajectory { poses }
        }
        other => return Err(format!("overlay: motion.mode {other:?}: expected off | trajectory | random | linear")),
    };
    let within = match &m.within_volume {
        None => None,
        Some(w) => {
            match shots_3d {
                None if mb <= 1 => {
                    return Err("overlay: motion.within_volume describes multiband shot events and needs \
                                MultibandAccelerationFactor > 1".to_string());
                }
                Some(s) if s <= 1 => {
                    return Err("overlay: motion.within_volume describes events between the shots of a segmented \
                                3D readout and needs NumberShots > 1".to_string());
                }
                _ => {}
            }
            for (name, v) in [("dropout_rate", w.dropout_rate), ("severity", w.severity)] {
                if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
                    return Err(format!("overlay: motion.within_volume.{name} must be in [0, 1], got {v}"));
                }
            }
            Some(WithinVolume {
                dropout_rate: w.dropout_rate,
                severity: w.severity,
                jump_mm: vec3_amplitude(w.jump_mm, "overlay motion.within_volume.jump_mm")?,
                jump_deg: vec3_amplitude(w.jump_deg, "overlay motion.within_volume.jump_deg")?,
            })
        }
    };
    if matches!(mode, MotionMode::Off) && within.is_none() {
        return Ok(None);
    }
    Ok(Some(MotionSpec { mode, mode_name, within }))
}

// ---------------------------------------------------------------- the parse

/// Parse from already-read text. [`load`] is the file-reading wrapper.
pub fn parse(sidecar: &Value, aslcontext: &str, overlay: Option<&Overlay>, phantom: Option<&PhantomParams>)
    -> Result<Protocol, String>
{
    parse_echoes(std::slice::from_ref(sidecar), aslcontext, overlay, phantom)
}

/// JSON equality with numbers compared as numbers (`60` equals `60.0`), elementwise through
/// arrays and objects.
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_json(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w)))
        }
        _ => match (a.as_f64(), b.as_f64()) {
            (Some(p), Some(q)) => p == q,
            _ => a == b,
        },
    }
}

/// Hadamard time encoding (P6 addendum, part A, "Inputs" and "Refusals"): the rows describe the
/// decoded dataset (`deltam` rows `H - 1` per cycle in sub-bolus order, `m0scan` rows only between
/// cycles); the sub-boli's durations are the same in every cycle, every row's PostLabelingDelay is
/// its sub-bolus's effective delay `PLD + sum_{k>j} tau_k`, and the repetition time, the
/// suppression pulse set and the crushing VENC are each one per cycle.
///
/// Under Look-Locker (P7 addendum, part B; `ll` is `Some((M, flips))`): each encoded preparation
/// is read `M` times, and the rows list the decoded volumes readout-major, `(H - 1) M` per cycle;
/// row `(j, n)`'s delay is `PLD_n + sum_{k>j} tau_k`, `PLD_n` strictly increasing; `FlipAngle` and
/// `VascularCrushingVENC` agree across the sub-boli of a readout (one VENC per readout index, not
/// per cycle); the cycle's pulse set is one and its pulses precede the first readout.
#[allow(clippy::too_many_arguments)]
fn hadamard_spec(
    ho: Option<&HadamardOverlay>, label_type: LabelType, rows: &[Row], pld: &[f64], suppression: Option<&SuppressionSpec>,
    crushing: Option<&CrushSpec>, compat: bool, ll: Option<(usize, Option<&[f64]>)>,
) -> Result<Option<HadamardSpec>, String> {
    let Some(ho) = ho else { return Ok(None) };
    let (m, flips) = ll.unwrap_or((1, None));
    if m == 0 {
        return Err("overlay: look_locker.readouts_per_cycle must be at least 1".to_string());
    }
    let order = ho.order.ok_or("overlay: [hadamard] needs order (4, 8, 16 or 32)")?;
    if !crate::hadamard::ORDERS.contains(&order) {
        return Err(format!("overlay: hadamard.order {order}: the Sylvester orders 4, 8, 16 and 32 are supported"));
    }
    if label_type == LabelType::Pasl {
        return Err("[hadamard] with PASL: time encoding switches the labeling during the bolus, which a pulsed \
                    label cannot; (P)CASL only".to_string());
    }
    if compat {
        return Err("[hadamard] under [compat] asldro = true: simasl has no time-encoded labeling".to_string());
    }
    if let Some(i) = rows.iter().position(|r| matches!(r.kind, RowKind::Control | RowKind::Label)) {
        return Err(format!(
            "aslcontext.tsv row {i} is {}: with [hadamard] the context lists the decoded volumes, deltam rows {} per \
             cycle and m0scan rows between cycles", rows[i].kind.as_str(), order - 1));
    }
    let n_sub = order - 1;
    // the decoded rows of one cycle: H - 1 per readout
    let per = n_sub * m;
    let mut cycles = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        if rows[i].kind == RowKind::M0scan {
            i += 1;
            continue;
        }
        let run = rows[i..].iter().take_while(|r| r.kind == RowKind::Deltam).count();
        if run % per != 0 {
            let what = if m > 1 {
                format!("{n_sub} sub-boli at each of {m} readouts (look_locker.readouts_per_cycle), {per} rows")
            } else {
                format!("{n_sub} sub-boli")
            };
            return Err(format!(
                "aslcontext.tsv rows {i}..{} are {run} deltam rows, not a whole number of order-{order} cycles of {what}; \
                 m0scan rows go only between cycles", i + run - 1));
        }
        for c in 0..run / per {
            cycles.push(HadamardCycle { rows: (i + c * per..i + (c + 1) * per).collect() });
        }
        i += run;
    }
    if cycles.is_empty() {
        return Err("[hadamard] with no deltam rows: nothing is encoded".to_string());
    }
    // the sub-boli from the first readout's rows (the only readout without Look-Locker)
    let tau: Vec<f64> = cycles[0].rows[..n_sub].iter().map(|&r| rows[r].tau).collect();
    // "sub-bolus j", and the readout under Look-Locker
    let at = |j: usize, n: usize| if m > 1 { format!("sub-bolus {}, readout {}", j + 1, n + 1) } else { format!("sub-bolus {}", j + 1) };
    for (c, cy) in cycles.iter().enumerate() {
        for (k, &r) in cy.rows.iter().enumerate() {
            let (j, n) = (k % n_sub, k / n_sub);
            if (rows[r].tau - tau[j]).abs() > 1e-9 {
                return Err(format!(
                    "LabelingDuration of row {r} (cycle {}, {}) is {} s, but {} s in cycle 1: the sub-boli \
                     are the same in every cycle", c + 1, at(j, n), rows[r].tau, tau[j]));
            }
        }
    }
    let tau_tot: f64 = tau.iter().sum();
    let mut spans = Vec::with_capacity(n_sub);
    let mut a = 0.0;
    for &t in &tau {
        spans.push((a, a + t));
        a += t;
    }
    // each readout's delay, from its last sub-bolus's row in cycle 1
    let plds: Vec<f64> = (0..m).map(|n| pld[cycles[0].rows[n * n_sub + n_sub - 1]]).collect();
    if let Some(n) = (1..m).find(|&n| plds[n] <= plds[n - 1]) {
        return Err(format!(
            "[hadamard] with LookLocker: true: readout {}'s delay PLD_{} = {} s does not follow readout {}'s {} s (the \
             last sub-bolus's PostLabelingDelay of each readout must strictly increase)", n + 1, n + 1, plds[n], n, plds[n - 1]));
    }
    let pld_n = plds[0];
    for (c, cy) in cycles.iter().enumerate() {
        for (k, &r) in cy.rows.iter().enumerate() {
            let (j, n) = (k % n_sub, k / n_sub);
            let want = plds[n] + tau[j + 1..].iter().sum::<f64>();
            if (pld[r] - want).abs() > 1e-6 {
                return Err(format!(
                    "PostLabelingDelay of row {r} (cycle {}, {}) is {} s; with [hadamard] each row's delay is \
                     its sub-bolus's effective delay PLD + the later sub-boli's durations = {want} s (PLD {} s, the \
                     last sub-bolus's)", c + 1, at(j, n), pld[r], plds[n]));
            }
        }
        let tr0 = rows[cy.rows[0]].tr;
        if let Some(&r) = cy.rows.iter().find(|&&r| rows[r].tr != tr0) {
            return Err(format!(
                "RepetitionTimePreparation of row {r} is {} s, but {tr0} s elsewhere in cycle {}: one repetition per cycle",
                rows[r].tr, c + 1));
        }
        if let Some(s) = suppression {
            let set = &s.per_row[cy.rows[0]];
            if cy.rows.iter().any(|&r| s.per_row[r] != *set) {
                return Err(format!(
                    "background suppression in cycle {}: its deltam rows map to different pulse sets (by their \
                     PostLabelingDelay), but a cycle's preparations share one; list the same set for its delays",
                    c + 1));
            }
            if let Some(&p) = set.iter().find(|&&p| p >= tau_tot + pld_n) {
                return Err(format!(
                    "background-suppression pulse at {p} s in cycle {} is at or after the readout at {} s (the encoded \
                     labeling's {tau_tot} s plus PLD {pld_n} s); P3 models pulses before the first excitation only",
                    c + 1, tau_tot + pld_n));
            }
            let early_ok = matches!(s.model, SuppressionModel::BolusPosition(r) if !matches!(r, Region::Global));
            if let Some(&p) = set.iter().find(|&&p| p < tau_tot && !early_ok) {
                return Err(format!(
                    "background-suppression pulse at {p} s falls inside the encoded labeling [0, {tau_tot}] s of cycle {}; \
                     a global pulse there inverts sub-boli and inflowing blood alike, which is not modeled (use \
                     pulse_region = \"slab\" under the bolus-position model)", c + 1));
            }
        }
        if let Some(cr) = crushing {
            if m == 1 {
                let v0 = cr.venc[cy.rows[0]];
                if cy.rows.iter().any(|&r| cr.venc[r] != v0) {
                    return Err(format!("VascularCrushingVENC varies within cycle {}: one VENC per cycle", c + 1));
                }
            } else {
                // P7 part A: under Look-Locker one VENC per readout index (each readout its own
                // bipolar gradients), the same for every sub-bolus of it
                for (k, &r) in cy.rows.iter().enumerate() {
                    let (j, n) = (k % n_sub, k / n_sub);
                    let v0 = cr.venc[cy.rows[n * n_sub]];
                    if cr.venc[r] != v0 {
                        return Err(format!(
                            "VascularCrushingVENC of row {r} (cycle {}, {}) is {}, but {v0} for sub-bolus 1 of the same \
                             readout: a readout's decoded volumes share its VENC", c + 1, at(j, n), cr.venc[r]));
                    }
                }
            }
        }
        if let Some(f) = flips {
            for (k, &r) in cy.rows.iter().enumerate() {
                let (j, n) = (k % n_sub, k / n_sub);
                let f0 = f[cy.rows[n * n_sub]];
                if f[r] != f0 {
                    return Err(format!(
                        "FlipAngle of row {r} (cycle {}, {}) is {} degrees, but {f0} for sub-bolus 1 of the same readout: \
                         a readout's decoded volumes share its excitation", c + 1, at(j, n), f[r]));
                }
            }
        }
    }
    let report_leakage = match ho.report_leakage {
        Some(b) => (b, Source::Overlay),
        None => (true, Source::Default),
    };
    Ok(Some(HadamardSpec { order, report_leakage, tau, spans, tau_tot, pld: pld_n, cycles, readouts: m, plds }))
}

/// What [`look_locker_spec`] reads.
struct LlInputs<'a> {
    on: bool,
    overlay: Option<&'a Overlay>,
    rows: &'a [Row],
    pld: &'a [f64],
    flips: Option<Vec<f64>>,
    contrast: Contrast,
    ge: Option<&'a GeSpec>,
    is_3d: bool,
    /// P7 part C: the 3D readout is the gradient-echo train, which Look-Locker can read.
    epi3d: bool,
    hadamard: Option<&'a HadamardSpec>,
    compat: bool,
    suppression: Option<&'a SuppressionSpec>,
    m0_type: M0Type,
}

/// Look-Locker readouts (P6 addendum, part B, "Inputs" and "Refusals"): gradient echo, 2D; the
/// cycles from the arrays (a maximal run of consecutive non-m0scan rows of one volume type with
/// strictly increasing PostLabelingDelay and equal RepetitionTimePreparation and
/// LabelingDuration; an m0scan row is its own cycle), checked against
/// `[look_locker] readouts_per_cycle`; one suppression pulse set per cycle; flips in (0, 90]; the
/// separate M0's flip from `[m0] flip_angle`, required with a FlipAngle array. The P4 parts
/// (exchange, the arterial compartment, crushing, bolus-position suppression) are allowed (P7
/// addendum, part A), with their own checks; crushing's VENC is per readout (per row).
fn look_locker_spec(i: LlInputs) -> Result<Option<LookLockerSpec>, String> {
    let llo = i.overlay.and_then(|o| o.look_locker.as_ref());
    let m0_flip = i.overlay.and_then(|o| o.m0.as_ref()).and_then(|m| m.flip_angle);
    if !i.on {
        if llo.is_some() {
            return Err("overlay: [look_locker] is read only with LookLocker: true".to_string());
        }
        if m0_flip.is_some() {
            return Err("overlay: m0.flip_angle is the separate M0 excitation of a Look-Locker series (LookLocker: \
                        true); otherwise the M0 takes the series' own".to_string());
        }
        return Ok(None);
    }
    for (what, on) in [
        ("MRAcquisitionType 3D with a GRASE or spiral readout (3D Look-Locker reads a gradient-echo train: [readout] \
          type = \"epi3d\", P7 part C)", i.is_3d && !i.epi3d),
        ("[compat] asldro = true", i.compat),
    ] {
        if on {
            return Err(format!("LookLocker: true with {what}: refused (P6 addendum, part B)"));
        }
    }
    if i.contrast != Contrast::GradientEcho {
        return Err("LookLocker: true needs [signal] acq_contrast = \"ge\": the readouts are low-flip gradient-echo \
                    excitations; a spin-echo Look-Locker series is refused".to_string());
    }
    let n = i.rows.len();
    let flip_array = i.flips.is_some();
    if !flip_array && i.ge.is_some_and(|g| g.flip == Source::Default) {
        return Err("LookLocker: true requires FlipAngle (the readouts' excitation, a scalar or one per volume); \
                    without it the gradient-echo default of 90 degrees would saturate every readout".to_string());
    }
    if flip_array && i.ge.is_some_and(|g| g.flip == Source::Overlay) {
        return Err("overlay: signal.excitation_flip_angle with a FlipAngle array: the array gives every volume its \
                    own excitation".to_string());
    }
    let flip_deg: Vec<f64> = match i.flips {
        Some(f) => f,
        None => vec![i.ge.map_or(90.0, |g| g.flip_deg); n],
    };
    if let Some((r, fa)) = flip_deg.iter().enumerate().find(|(_, &fa)| !(fa > 0.0 && fa <= 90.0)) {
        return Err(format!("FlipAngle {fa} for row {r}: a Look-Locker excitation is in (0, 90] degrees"));
    }
    let mut cycles = Vec::new();
    // P7 part B: under [hadamard] the context lists decoded volumes, so the cycles are not runs of
    // the arrays: each encoding cycle is represented by sub-bolus 1 of each readout (its row's
    // time is the readout's, tau_tot + PLD_n), each m0scan row by its own cycle;
    // hadamard_spec checked the readout structure
    if let Some(h) = i.hadamard {
        let mut v = 0;
        while v < n {
            if i.rows[v].kind == RowKind::M0scan {
                cycles.push(LookLockerCycle { rows: vec![v], m0scan: true });
                v += 1;
            } else {
                let cy = h.cycles.iter().find(|c| c.rows[0] == v).expect("hadamard_spec covers every deltam row");
                cycles.push(LookLockerCycle { rows: (0..h.readouts).map(|nn| h.row(cy, 0, nn)).collect(), m0scan: false });
                v += cy.rows.len();
            }
        }
    }
    let mut v = if i.hadamard.is_some() { n } else { 0 };
    while v < n {
        if i.rows[v].kind == RowKind::M0scan {
            cycles.push(LookLockerCycle { rows: vec![v], m0scan: true });
            v += 1;
            continue;
        }
        let mut end = v + 1;
        while end < n {
            let (a, b) = (&i.rows[end - 1], &i.rows[end]);
            if b.kind != a.kind || i.pld[end] <= i.pld[end - 1] || b.tr != a.tr || b.tau != a.tau {
                break;
            }
            end += 1;
        }
        cycles.push(LookLockerCycle { rows: (v..end).collect(), m0scan: false });
        v = end;
    }
    let readouts_per_cycle = match llo.and_then(|o| o.readouts_per_cycle) {
        Some(m) => {
            if let Some(c) = cycles.iter().find(|c| !c.m0scan && c.rows.len() != m) {
                return Err(format!(
                    "overlay: look_locker.readouts_per_cycle = {m}, but the cycle starting at row {} has {} readouts (a \
                     cycle is a run of one volume type with increasing PostLabelingDelay and equal \
                     RepetitionTimePreparation and LabelingDuration)", c.rows[0], c.rows.len()));
            }
            Some(m)
        }
        None => None,
    };
    if let Some(s) = i.suppression {
        for c in cycles.iter().filter(|c| !c.m0scan) {
            let set = &s.per_row[c.rows[0]];
            if c.rows.iter().any(|&r| s.per_row[r] != *set) {
                return Err(format!(
                    "background suppression in the Look-Locker cycle starting at row {}: its readouts map to different \
                     pulse sets (by their PostLabelingDelay), but a cycle has one preparation", c.rows[0]));
            }
        }
    }
    let m0_flip_deg = match (i.m0_type, m0_flip) {
        (M0Type::Separate, Some(fa)) => {
            if !(fa > 0.0 && fa <= 90.0) {
                return Err(format!("overlay: m0.flip_angle {fa}: an excitation in (0, 90] degrees"));
            }
            Some((fa, Source::Overlay))
        }
        (M0Type::Separate, None) if flip_array => {
            return Err("a Look-Locker series with a FlipAngle array needs overlay m0.flip_angle for its separate M0 \
                        (the array gives the series' volumes, not the M0 scan's)".to_string())
        }
        (M0Type::Separate, None) => Some((flip_deg[0], Source::Sidecar)),
        (_, Some(_)) => return Err("overlay: m0.flip_angle describes a separate M0 scan (M0Type \"Separate\")".to_string()),
        (_, None) => None,
    };
    Ok(Some(LookLockerSpec { cycles, flip_deg, flip_array, readouts_per_cycle, m0_flip_deg }))
}

/// The excitation timing of a Look-Locker series on its acquired grid (P6 addendum, part B,
/// "Timing on explicit intervals"), in seconds: slice excitations at `t_n + offset` sample
/// `[e + TE - h, e + TE + h]`; excitation groups in chronological order, each group's block ending
/// before the next group's excitation, a readout's last group before the next readout's first,
/// the last readout's before TR, every sample after its excitation; an m0scan cycle is excited at
/// the start of its repetition; the separate M0 at its own repetition time.
fn check_look_locker_timing(p: &Protocol, ll: &LookLockerSpec, h: f64) -> Result<(), String> {
    // P7 part B: with several echoes per readout the first bounds the first sample and the last
    // every check between excitations; gradient-echo blocks within one excitation must not overlap
    let tes = &p.echo_times_s;
    let te = tes[0];
    if te - h < 0.0 {
        return Err(format!("EchoTime {te} s samples from {} s before its excitation (half readout {h} s)", h - te));
    }
    for e in 1..tes.len() {
        if tes[e] - h < tes[e - 1] + h {
            return Err(format!(
                "gradient echo: echo {} samples [{}, {}] s after each Look-Locker excitation and echo {} [{}, {}] s; the \
                 readouts overlap (TotalReadoutTime {} s)", e, tes[e - 1] - h, tes[e - 1] + h, e + 1, tes[e] - h,
                tes[e] + h, p.total_readout_time_s));
        }
    }
    let te_last = tes[tes.len() - 1];
    let which = if tes.len() > 1 { format!(" (echo {}, the last)", tes.len()) } else { String::new() };
    let mut groups: Vec<f64> = p.slice_offsets.clone();
    groups.sort_by(f64::total_cmp);
    groups.dedup();
    let g_last = groups.last().copied().unwrap_or(0.0);
    let end_of = |e: f64| e + te_last + h;
    for c in &ll.cycles {
        let times: Vec<f64> = if c.m0scan { vec![0.0] } else { c.rows.iter().map(|&r| p.rows[r].t).collect() };
        let tr = p.rows[c.rows[0]].tr;
        for (n, &t) in times.iter().enumerate() {
            for w in groups.windows(2) {
                if end_of(t + w[0]) > t + w[1] {
                    return Err(format!(
                        "row {}: the slices excited at {} s read until {} s{which}, after the next slices' excitation at {} s",
                        c.rows[n.min(c.rows.len() - 1)], t + w[0], end_of(t + w[0]), t + w[1]));
                }
            }
            if let Some(&next) = times.get(n + 1) {
                if end_of(t + g_last) > next + groups.first().copied().unwrap_or(0.0) {
                    return Err(format!(
                        "row {}: the readout at {t} s reads its last slices until {} s{which}, after the next readout's \
                         excitation at {next} s", c.rows[n], end_of(t + g_last)));
                }
            }
        }
        let last = times[times.len() - 1];
        if end_of(last + g_last) > tr {
            return Err(format!(
                "row {}: the last readout's last slices read until {} s{which}, after the cycle's RepetitionTimePreparation {tr} s",
                c.rows[c.rows.len() - 1], end_of(last + g_last)));
        }
    }
    if let (M0Type::Separate, Some(tr)) = (p.m0_type, p.m0_repetition_time_s) {
        if end_of(g_last) > tr {
            return Err(format!("the separate M0 reads its last slices until {} s, after its repetition time {tr} s", end_of(g_last)));
        }
    }
    Ok(())
}

/// The echo times of a multi-TE protocol (P6 addendum, part C): one sidecar per echo, each with
/// a scalar `EchoTime`, strictly increasing, and every other key the same in all of them.
fn echo_times_of(sidecars: &[Value]) -> Result<Vec<f64>, String> {
    let mut tes = Vec::with_capacity(sidecars.len());
    for (e, s) in sidecars.iter().enumerate() {
        let te = match s.get("EchoTime") {
            Some(Value::Number(n)) => n.as_f64().unwrap(),
            Some(Value::Array(_)) => {
                return Err(format!(
                    "asl.json of echo {}: EchoTime is an array; with one sidecar per echo (the BIDS echo-N layout) \
                     each sidecar gives its own echo's EchoTime as a number", e + 1))
            }
            Some(_) => return Err(format!("asl.json of echo {}: EchoTime must be a number", e + 1)),
            None => return Err(format!("asl.json of echo {}: missing required field \"EchoTime\"", e + 1)),
        };
        tes.push(require_finite_positive(te, &format!("asl.json of echo {}: EchoTime", e + 1))?);
    }
    for (e, w) in tes.windows(2).enumerate() {
        if w[1] <= w[0] {
            return Err(format!(
                "the --asl-json sidecars are the echoes in order, so EchoTime must increase strictly: echo {} has {} s, \
                 echo {} has {} s", e + 1, w[0], e + 2, w[1]));
        }
    }
    let first = sidecars[0].as_object().ok_or("asl.json of echo 1 is not a JSON object")?;
    for (e, s) in sidecars.iter().enumerate().skip(1) {
        let other = s.as_object().ok_or_else(|| format!("asl.json of echo {} is not a JSON object", e + 1))?;
        let mut keys: Vec<&String> = first.keys().chain(other.keys()).filter(|k| k.as_str() != "EchoTime").collect();
        keys.sort();
        keys.dedup();
        for k in keys {
            let show = |v: Option<&Value>| v.map_or("absent".to_string(), |v| v.to_string());
            let (a, b) = (first.get(k), other.get(k));
            if !matches!((a, b), (Some(x), Some(y)) if same_json(x, y)) {
                return Err(format!(
                    "the echo sidecars must agree on every key but EchoTime: {k} is {} in echo 1 and {} in echo {}",
                    show(a), show(b), e + 1));
            }
        }
    }
    Ok(tes)
}

/// [`parse`] for one sidecar per echo, in echo order (P6 addendum, part C). One sidecar is
/// [`parse`] itself.
pub fn parse_echoes(sidecars: &[Value], aslcontext: &str, overlay: Option<&Overlay>, phantom: Option<&PhantomParams>)
    -> Result<Protocol, String>
{
    let sidecar = sidecars.first().ok_or("no ASL sidecar")?;
    let multi_echo = sidecars.len() > 1;
    let kinds = parse_aslcontext(aslcontext)?;
    let n = kinds.len();

    // Labeling scheme.
    let label_type = match string(sidecar, "ArterialSpinLabelingType")?.to_ascii_uppercase().as_str() {
        "PASL" => LabelType::Pasl,
        "CASL" => LabelType::Casl,
        "PCASL" => LabelType::Pcasl,
        other => return Err(format!("asl.json: ArterialSpinLabelingType {other:?} is not PASL, CASL or PCASL")),
    };

    // M0Type versus the rows.
    let m0_type = match string(sidecar, "M0Type")?.as_str() {
        "Included" => M0Type::Included,
        "Separate" => M0Type::Separate,
        "Estimate" => M0Type::Estimate,
        "Absent" => M0Type::Absent,
        other => return Err(format!("asl.json: M0Type {other:?} is not Included, Separate, Estimate or Absent")),
    };
    let m0_rows: Vec<usize> = kinds.iter().enumerate().filter(|(_, k)| **k == RowKind::M0scan).map(|(i, _)| i).collect();
    match m0_type {
        M0Type::Included if m0_rows.is_empty() => {
            return Err("asl.json: M0Type is \"Included\" but aslcontext.tsv has no m0scan row".to_string())
        }
        M0Type::Separate | M0Type::Estimate | M0Type::Absent if !m0_rows.is_empty() => {
            return Err(format!(
                "asl.json: M0Type is {:?} but aslcontext.tsv row {} is m0scan",
                string(sidecar, "M0Type")?, m0_rows[0]))
        }
        _ => {}
    }

    // The readout's dimensionality: 2D EPI, or a 3D segmented spin-echo train (P5 part B).
    let mr_type = string(sidecar, "MRAcquisitionType")?;
    let is_3d = match mr_type.as_str() {
        "2D" => false,
        "3D" => true,
        other => return Err(format!("asl.json: MRAcquisitionType {other:?}: 2D or 3D readouts only")),
    };
    let ro = overlay.and_then(|o| o.readout.as_ref());
    if !is_3d && ro.is_some() {
        return Err("overlay: [readout] describes a 3D echo train; MRAcquisitionType is \"2D\"".to_string());
    }

    // Things P1 does not model must not be silently simulated as if absent.
    // P6 part B: Look-Locker readouts (its rules are look_locker_spec's, once the rest is known)
    let look_locker = opt_bool(sidecar, "LookLocker")?;
    // (VascularCrushing is P4's, part C, resolved with the other P4 inputs below.)
    // Required by BIDS, and read rather than defaulted: an absent field would be written back
    // absent and the dataset would not validate.
    let background_suppression = match field(sidecar, "BackgroundSuppression")? {
        Value::Bool(b) => *b,
        _ => return Err("asl.json: BackgroundSuppression must be a boolean".to_string()),
    };
    if m0_type == M0Type::Estimate {
        num(sidecar, "M0Estimate").map_err(|_| "asl.json: M0Type \"Estimate\" requires a numeric M0Estimate".to_string())?;
    }

    // Timing per row. PLD is per volume with zeros on m0scan rows.
    let pld = per_row(num_or_array(sidecar, "PostLabelingDelay")?, "PostLabelingDelay", &kinds, true)?;
    let tau: Vec<f64> = match label_type {
        LabelType::Casl | LabelType::Pcasl => {
            per_row(num_or_array(sidecar, "LabelingDuration")?, "LabelingDuration", &kinds, true)?
        }
        LabelType::Pasl => {
            if !opt_bool(sidecar, "BolusCutOffFlag")? {
                return Err("asl.json: PASL requires BolusCutOffFlag true; without a bolus cutoff the \
                            kinetic model has no bolus duration".to_string());
            }
            string(sidecar, "BolusCutOffTechnique")
                .map_err(|_| "asl.json: BolusCutOffFlag true requires BolusCutOffTechnique".to_string())?;
            let (cut, _) = num_or_array(sidecar, "BolusCutOffDelayTime")?;
            if cut.len() > 2 {
                return Err(format!(
                    "asl.json: BolusCutOffDelayTime has {} entries; a number, or the first and last pulse \
                     times for Q2TIPS, are supported", cut.len()));
            }
            if cut.iter().any(|v| *v < 0.0) || cut.windows(2).any(|w| w[1] < w[0]) {
                return Err("asl.json: BolusCutOffDelayTime must be non-negative and non-decreasing".to_string());
            }
            // The bolus duration is the first cutoff time (Q2TIPS gives first and last).
            kinds.iter().map(|k| if *k == RowKind::M0scan { 0.0 } else { cut[0] }).collect()
        }
    };
    let tr = per_row(num_or_array(sidecar, "RepetitionTimePreparation")?, "RepetitionTimePreparation", &kinds, false)?;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        require_finite_positive(tr[i], &format!("asl.json: RepetitionTimePreparation for row {i}"))?;
        if kinds[i] != RowKind::M0scan {
            require_finite_nonneg(pld[i], &format!("asl.json: PostLabelingDelay for row {i}"))?;
            require_finite_positive(tau[i], &format!("asl.json: bolus duration for row {i}"))?;
        }
        let t = match (kinds[i], label_type) {
            (RowKind::M0scan, _) => 0.0,
            (_, LabelType::Pasl) => {
                if pld[i] <= tau[i] {
                    return Err(format!(
                        "asl.json: PASL row {i}: PostLabelingDelay {} <= BolusCutOffDelayTime {}; the cutoff \
                         pulse would fall after the readout", pld[i], tau[i]));
                }
                pld[i]
            }
            (_, _) => pld[i] + tau[i],
        };
        rows.push(Row { kind: kinds[i], t, tau: tau[i], tr: tr[i] });
    }

    // Readout geometry and timing. In 3D every partition of a shot shares the slab excitation,
    // to which BIDS's PostLabelingDelay runs, so there is no slice timing (BIDS: SliceTiming must
    // not be defined for 3D), no slice offsets, and no multiband.
    let (slice_offsets, timing, max_offset, mb, mb_interleaved) = if is_3d {
        for key in ["SliceTiming", "SliceEncodingDirection"] {
            if sidecar.get(key).is_some_and(|v| !v.is_null()) {
                return Err(format!(
                    "asl.json: {key} with MRAcquisitionType \"3D\": every partition shares the slab excitation \
                     (BIDS: SliceTiming must not be defined for 3D)"));
            }
        }
        if let Some(v) = opt_num(sidecar, "MultibandAccelerationFactor")? {
            if v != 1.0 {
                return Err(format!("asl.json: MultibandAccelerationFactor {v} with MRAcquisitionType \"3D\": a 3D \
                                    readout excites one slab"));
            }
        }
        (Vec::new(), Vec::new(), 0.0, 1usize, false)
    } else {
    let (timing, _) = num_or_array(sidecar, "SliceTiming")?;
    for t in &timing {
        require_finite_nonneg(*t, "asl.json: SliceTiming entry")?;
    }
    let reversed = match sidecar.get("SliceEncodingDirection").and_then(Value::as_str) {
        None | Some("k") => false,
        Some("k-") => true,
        Some(other) => {
            return Err(format!("asl.json: SliceEncodingDirection {other:?}: the slice axis must be k or k-"))
        }
    };
    let tmin = timing.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut slice_offsets: Vec<f64> = timing.iter().map(|t| t - tmin).collect();
    if reversed {
        slice_offsets.reverse();
    }
    // Every slice's readout must fall inside the repetition, or the tissue has a negative
    // recovery interval (P3 addendum, timing constraints; true for P1 protocols as well).
    let max_offset = slice_offsets.iter().cloned().fold(0.0, f64::max);
    for (i, r) in rows.iter().enumerate() {
        // m0scan rows have t = 0: their readout starts at the excitation and spans the offsets
        if r.t + max_offset > r.tr {
            return Err(format!(
                "asl.json: row {i} reads its last slice at {} s (signal time {} s + slice offset {} s), after \
                 its RepetitionTimePreparation {} s", r.t + max_offset, r.t, max_offset, r.tr));
        }
    }
    let mb = match opt_num(sidecar, "MultibandAccelerationFactor")? {
        None => 1,
        Some(v) if v >= 1.0 && v.fract() == 0.0 => v as usize,
        Some(v) => return Err(format!("asl.json: MultibandAccelerationFactor {v} is not a positive integer")),
    };
    if mb > 1 {
        // mb slices share each excitation, so every distinct timing must occur exactly mb times.
        let mut sorted: Vec<u64> = timing.iter().map(|t| t.to_bits()).collect();
        sorted.sort_unstable();
        let mut i = 0;
        while i < sorted.len() {
            let j = sorted[i..].iter().take_while(|b| **b == sorted[i]).count();
            if j != mb {
                return Err(format!(
                    "asl.json: MultibandAccelerationFactor {mb} but slice time {} is shared by {j} slices",
                    f64::from_bits(sorted[i])));
            }
            i += j;
        }
    }
    let mb_interleaved = if mb > 1 { multiband_schedule(&slice_offsets, mb)? } else { false };
    (slice_offsets, timing, max_offset, mb, mb_interleaved)
    };

    // The 3D readout kind: the overlay's [readout] type over PulseSequenceType (P5 part B).
    let readout_kind = if is_3d {
        let (name, src) = match (ro.and_then(|r| r.kind.as_deref()), sidecar.get("PulseSequenceType").and_then(Value::as_str)) {
            (Some(k), _) => (k.to_string(), Source::Overlay),
            (None, Some(k)) => (k.to_string(), Source::Sidecar),
            (None, None) => (String::new(), Source::Default),
        };
        let lower = name.to_ascii_lowercase();
        let kind = if lower.contains("grase") {
            ReadoutKind::Grase
        } else if lower.contains("spiral") {
            ReadoutKind::Spiral
        } else if lower.contains("epi") {
            // P7 part C: the gradient-echo stack-of-EPI train ("epi3d", "3D EPI")
            ReadoutKind::Epi3d
        } else {
            return Err(format!(
                "3D readout {name:?} (PulseSequenceType, or [readout] type): the 3D readouts are GRASE (\"grase\"), \
                 the stack of spirals (\"spiral\") and the gradient-echo stack of EPI (\"epi3d\"); [readout] type \
                 selects one"));
        };
        Some((kind, src))
    } else {
        None
    };

    let side_ped = sidecar.get("PhaseEncodingDirection").and_then(Value::as_str).map(str::to_string);
    let ro_ped = ro.and_then(|r| r.phase_encoding_direction.clone());
    let ped = match readout_kind.map(|k| k.0) {
        None => string(sidecar, "PhaseEncodingDirection")?,
        // GRASE: required (it sets the traversal and the distortion direction), overlay first
        Some(ReadoutKind::Grase | ReadoutKind::Epi3d) => ro_ped.or(side_ped).ok_or(
            "a GRASE or 3D EPI readout needs a phase-encode direction (it sets the traversal and the distortion direction): \
             the sidecar has no PhaseEncodingDirection; give [readout] phase_encoding_direction = \"j\" or \"j-\""
                .to_string())?,
        Some(ReadoutKind::Spiral) => {
            if sidecar.get("PhaseEncodingDirection").is_some_and(|v| !v.is_null()) || ro_ped.is_some() {
                return Err("PhaseEncodingDirection with a spiral readout: a spiral has no phase-encode axis".to_string());
            }
            String::new()
        }
    };
    let reverse_phase = match (ped.as_str(), readout_kind.map(|k| k.0)) {
        ("j-", _) => false,
        ("j", _) => true,
        ("", Some(ReadoutKind::Spiral)) => false,
        (_, None) => {
            return Err(format!(
                "asl.json: PhaseEncodingDirection {ped:?}: P1 encodes phase along the second data axis \
                 (j or j-) only"))
        }
        _ => {
            return Err(format!(
                "PhaseEncodingDirection {ped:?}: phase is encoded along the second data axis (j or j-) only"))
        }
    };
    let echo_times_s = if multi_echo {
        // P7 part C: a 3D gradient-echo train reads several echoes per excitation
        if is_3d && readout_kind.is_none_or(|k| k.0 != ReadoutKind::Epi3d) {
            return Err("multi-TE (one --asl-json per echo) with a 3D GRASE or spiral readout: multi-echo spin-echo \
                        trains are deferred (P7 addendum); the 3D gradient-echo train (\"epi3d\") reads several echoes"
                .to_string());
        }
        echo_times_of(sidecars)?
    } else {
        let (te, _) = num_or_array(sidecar, "EchoTime")?;
        if te.iter().any(|v| *v != te[0]) {
            return Err(format!(
                "asl.json: EchoTime array has unequal entries {te:?}; multi-TE ASL takes one sidecar per echo (repeat \
                 --asl-json), written as the BIDS echo-N layout"));
        }
        vec![require_finite_positive(te[0], "asl.json: EchoTime")?]
    };
    let echo_time_s = echo_times_s[0];
    let last_echo_time_s = echo_times_s[echo_times_s.len() - 1];
    // 2D: required. 3D GRASE: one of the sources of the line spacing (P5 part B, "Timing from
    // BIDS"), optional; spirals have no phase-encode readout, so it is refused there.
    let total_readout_time_s = match readout_kind.map(|k| k.0) {
        None => require_finite_positive(num(sidecar, "TotalReadoutTime")?, "asl.json: TotalReadoutTime")?,
        Some(ReadoutKind::Grase | ReadoutKind::Epi3d) => match opt_num(sidecar, "TotalReadoutTime")? {
            Some(v) => require_finite_positive(v, "asl.json: TotalReadoutTime")?,
            None => 0.0,
        },
        Some(ReadoutKind::Spiral) => {
            for key in ["TotalReadoutTime", "EffectiveEchoSpacing"] {
                if sidecar.get(key).is_some_and(|v| !v.is_null()) {
                    return Err(format!("asl.json: {key} with a spiral readout: it describes a phase-encode readout"));
                }
            }
            0.0
        }
    };
    let field_strength = require_finite_positive(num(sidecar, "MagneticFieldStrength")?, "asl.json: MagneticFieldStrength")?;
    let (vs, _) = num_or_array(sidecar, "AcquisitionVoxelSize")?;
    if vs.len() != 3 || vs.iter().any(|v| *v <= 0.0) {
        return Err("asl.json: AcquisitionVoxelSize must be three positive numbers".to_string());
    }
    let voxel_size_mm = [vs[0], vs[1], vs[2]];
    let accel = match opt_num(sidecar, "ParallelReductionFactorInPlane")? {
        None => 1,
        Some(v) if v >= 1.0 && v.fract() == 0.0 => v as usize,
        Some(v) => return Err(format!("asl.json: ParallelReductionFactorInPlane {v} is not a positive integer")),
    };

    // Phantom consistency.
    if let Some(pf) = phantom.and_then(|p| p.field_strength) {
        if (pf - field_strength).abs() > 1e-9 {
            return Err(format!(
                "phantom.json MagneticFieldStrength {pf} disagrees with asl.json MagneticFieldStrength \
                 {field_strength}; the phantom's relaxation values are field-specific"));
        }
    }

    // Kinetic constants: overlay > sidecar (alpha only) > phantom > default.
    let ko = overlay.and_then(|o| o.kinetic.as_ref());
    if let Some(k) = ko {
        for (name, v) in [("label_efficiency", k.label_efficiency), ("lambda_blood_brain", k.lambda_blood_brain),
                          ("t1_arterial_blood", k.t1_arterial_blood)] {
            if let Some(v) = v {
                require_finite_nonneg(v, &format!("overlay kinetic.{name}"))?;
            }
        }
    }
    let alpha = if let Some(v) = ko.and_then(|k| k.label_efficiency) {
        (v, Source::Overlay)
    } else if let Some(v) = opt_num(sidecar, "LabelingEfficiency")? {
        (v, Source::Sidecar)
    } else {
        // Alsop et al. 2015 consensus values; the spec lists PCASL and PASL, CASL follows the
        // same source.
        let d = match label_type {
            LabelType::Pcasl => 0.85,
            LabelType::Pasl => 0.98,
            LabelType::Casl => 0.68,
        };
        (d, Source::Default)
    };
    let lambda = if let Some(v) = ko.and_then(|k| k.lambda_blood_brain) {
        (v, Source::Overlay)
    } else if let Some(v) = phantom.and_then(|p| p.lambda) {
        (v, Source::Phantom)
    } else {
        (0.9, Source::Default)
    };
    let by_field = |name: &str, at3: f64, at15: f64| -> Result<f64, String> {
        if (field_strength - 3.0).abs() < 1e-9 {
            Ok(at3)
        } else if (field_strength - 1.5).abs() < 1e-9 {
            Ok(at15)
        } else {
            Err(format!("no default {name} at {field_strength} T; set it in the overlay"))
        }
    };
    let t1b = if let Some(v) = ko.and_then(|k| k.t1_arterial_blood) {
        (v, Source::Overlay)
    } else if let Some(v) = phantom.and_then(|p| p.t1b) {
        (v, Source::Phantom)
    } else {
        (by_field("t1_arterial_blood", 1.65, 1.35)?, Source::Default)
    };
    let so = overlay.and_then(|o| o.signal.as_ref());
    let t2_blood_s = if let Some(v) = so.and_then(|s| s.t2_blood) {
        (require_finite_positive(v, "overlay signal.t2_blood")?, Source::Overlay)
    } else {
        (by_field("t2_blood", 0.165, 0.290)?, Source::Default)
    };
    let contrast = parse_contrast(so.and_then(|s| s.acq_contrast.as_deref()).unwrap_or("se"))?;
    if look_locker && contrast != Contrast::GradientEcho && !is_3d {
        return Err("LookLocker: true needs [signal] acq_contrast = \"ge\": the readouts are low-flip gradient-echo \
                    excitations; a spin-echo Look-Locker series is refused".to_string());
    }

    // Inversion recovery (P3 addendum, part B): InversionTime and FlipAngle are standard BIDS
    // fields, so they are inputs with the usual precedence; the inversion angle is overlay-only.
    let side_ti = opt_num(sidecar, "InversionTime")?;
    // BIDS puts FlipAngle in [0, 360]; simasl's signed [-180, 180] is the internal convention,
    // so 330 becomes -30 (the writer does the reverse).
    // P6 part B: a FlipAngle array is one excitation per volume, which only Look-Locker reads
    let ll_flips: Option<Vec<f64>> = match sidecar.get("FlipAngle") {
        Some(Value::Array(_)) if look_locker => Some(per_row(num_or_array(sidecar, "FlipAngle")?, "FlipAngle", &kinds, false)?),
        Some(Value::Array(_)) => {
            return Err("asl.json: FlipAngle is an array, one excitation per volume, which only a Look-Locker series \
                        (LookLocker: true) has".to_string())
        }
        _ => None,
    };
    let side_fa = match ll_flips.as_ref().map(|f| {
        (0..f.len()).find(|&i| kinds[i] != RowKind::M0scan).map_or(f[0], |i| f[i])
    }) {
        Some(first) => Some(first),
        None => match opt_num(sidecar, "FlipAngle")? {
        Some(v) => {
            if !(v.is_finite() && (0.0..=360.0).contains(&v)) {
                return Err(format!("asl.json: FlipAngle must be in [0, 360] degrees, got {v}"));
            }
            Some(if v > 180.0 { v - 360.0 } else { v })
        }
        None => None,
        },
    };
    // The 3D readouts are spin-echo trains (P5 part B): a 3D gradient-echo readout and 3D inversion
    // recovery are deferred, and the sidecar's FlipAngle is the refocusing angle there (read below),
    // not an excitation the spin-echo rule would hold to 90.
    let epi3d = readout_kind.is_some_and(|k| k.0 == ReadoutKind::Epi3d);
    if epi3d {
        // P7 part C: the 3D EPI train is a gradient-echo readout
        if contrast != Contrast::GradientEcho {
            return Err(format!(
                "[readout] type \"epi3d\" (a 3D gradient-echo train) needs [signal] acq_contrast = \"ge\", not {:?}; \
                 the 3D spin-echo readouts are \"grase\" and \"spiral\"", contrast.as_str()));
        }
    } else if is_3d {
        match contrast {
            Contrast::SpinEcho => {}
            Contrast::GradientEcho => return Err(
                "acq_contrast \"ge\" with MRAcquisitionType \"3D\" and a GRASE or spiral readout: the 3D gradient-echo \
                 readout is [readout] type = \"epi3d\" (P7)".to_string()),
            Contrast::InversionRecovery => return Err(
                "acq_contrast \"ir\" with MRAcquisitionType \"3D\": inversion recovery composed with an echo train \
                 is deferred".to_string()),
        }
    }
    // under GRASE and spiral the sidecar's FlipAngle is the refocusing angle; under "epi3d" the
    // excitation (P7 part C)
    let side_fa_excitation = if is_3d && !epi3d { None } else { side_fa };
    let (ir, ge) = match contrast {
        Contrast::SpinEcho => {
            for (what, fa) in [("asl.json: FlipAngle", side_fa_excitation), ("overlay: signal.excitation_flip_angle", so.and_then(|s| s.excitation_flip_angle))] {
                if let Some(fa) = fa {
                    if (fa - 90.0).abs() > 1e-9 {
                        return Err(format!(
                            "{what} {fa} with acq_contrast \"se\": the spin-echo signal equation assumes a 90-degree \
                             excitation; an inversion-recovery readout takes acq_contrast = \"ir\""));
                    }
                }
            }
            if side_ti.is_some() || so.and_then(|s| s.inversion_time).is_some() {
                return Err("InversionTime with acq_contrast \"se\": no inversion is simulated, and echoing the \
                            field would describe a preparation that did not happen; use acq_contrast = \"ir\""
                    .to_string());
            }
            if so.and_then(|s| s.inversion_flip_angle).is_some() {
                return Err("overlay: signal.inversion_flip_angle is undefined for acq_contrast \"se\"".to_string());
            }
            (None, None)
        }
        Contrast::InversionRecovery => {
            if background_suppression {
                return Err("acq_contrast \"ir\" with BackgroundSuppression true: the IR equation is a steady state \
                            and the suppression model is a timeline; P3 does not compose them".to_string());
            }
            let (ti, ti_src) = match (so.and_then(|s| s.inversion_time), side_ti) {
                (Some(v), _) => (v, Source::Overlay),
                (None, Some(v)) => (v, Source::Sidecar),
                (None, None) => (1.0, Source::Default),
            };
            let (fa, fa_src) = match (so.and_then(|s| s.excitation_flip_angle), side_fa) {
                (Some(v), _) => (v, Source::Overlay),
                (None, Some(v)) => (v, Source::Sidecar),
                (None, None) => (90.0, Source::Default),
            };
            let (fi, fi_src) = match so.and_then(|s| s.inversion_flip_angle) {
                Some(v) => (v, Source::Overlay),
                None => (180.0, Source::Default),
            };
            require_finite_nonneg(ti, "inversion time")?;
            for (name, v) in [("excitation flip angle", fa), ("inversion flip angle", fi)] {
                if !(v.is_finite() && (-180.0..=180.0).contains(&v)) {
                    return Err(format!("{name} must be in [-180, 180] degrees, got {v}"));
                }
            }
            for (i, r) in rows.iter().enumerate() {
                if r.kind != RowKind::M0scan && r.tr < last_echo_time_s + ti {
                    return Err(format!(
                        "asl.json: row {i}: RepetitionTimePreparation {} s is shorter than EchoTime {} s + inversion \
                         time {ti} s (simasl's IR constraint)", r.tr, last_echo_time_s));
                }
            }
            (Some(IrSpec {
                params: IrParams { inversion_time: ti, excitation_flip_deg: fa, inversion_flip_deg: fi },
                inversion_time: ti_src, excitation_flip: fa_src, inversion_flip: fi_src,
            }), None)
        }
        // Gradient echo (P5 addendum, part A): the excitation angle is overlay over sidecar over
        // 90; no inversion is simulated, so its fields are refused as under "se".
        Contrast::GradientEcho => {
            if side_ti.is_some() || so.and_then(|s| s.inversion_time).is_some() {
                return Err("InversionTime with acq_contrast \"ge\": no inversion is simulated, and echoing the \
                            field would describe a preparation that did not happen".to_string());
            }
            if so.and_then(|s| s.inversion_flip_angle).is_some() {
                return Err("overlay: signal.inversion_flip_angle is undefined for acq_contrast \"ge\"".to_string());
            }
            let (fa, fa_src) = match (so.and_then(|s| s.excitation_flip_angle), side_fa) {
                (Some(v), _) => (v, Source::Overlay),
                (None, Some(v)) => (v, Source::Sidecar),
                (None, None) => (90.0, Source::Default),
            };
            if !(fa.is_finite() && (-180.0..=180.0).contains(&fa)) {
                return Err(format!("excitation flip angle must be in [-180, 180] degrees, got {fa}"));
            }
            (None, Some(GeSpec { flip_deg: fa, flip: fa_src }))
        }
    };

    // The 3D readout as parsed (P5 part B, "Inputs"); what needs the grid is resolve_readout's.
    let readout = match readout_kind {
        None => None,
        Some((kind, kind_src)) => {
            let empty = ReadoutOverlay::default();
            let r = ro.unwrap_or(&empty);
            let grase_only = [("ky_segments", r.ky_segments.is_some()), ("line_spacing", r.line_spacing.is_some()),
                              ("readout_samples", r.readout_samples.is_some())];
            let spiral_only = [("interleaves", r.interleaves.is_some()),
                               ("spiral_readout_time", r.spiral_readout_time.is_some()), ("dwell_time", r.dwell_time.is_some()),
                               ("radial_oversampling", r.radial_oversampling.is_some())];
            // P7 part C: the spin-echo train's keys and the 3D EPI train's
            let echo_train_only = [("echo_spacing", r.echo_spacing.is_some()), ("refocusing_time", r.refocusing_time.is_some()),
                                   ("refocusing_flip_angle", r.refocusing_flip_angle.is_some())];
            let epi3d_only = [("excitation_spacing", r.excitation_spacing.is_some()),
                              ("excitation_time", r.excitation_time.is_some()),
                              ("slab_entry_time", r.slab_entry_time.is_some()), ("max_t1_groups", r.max_t1_groups.is_some()),
                              ("node_tolerance", r.node_tolerance.is_some())];
            let wrong: Vec<(&str, bool, &str)> = match kind {
                ReadoutKind::Grase => spiral_only.iter().map(|&(k, on)| (k, on, "spiral"))
                    .chain(epi3d_only.iter().map(|&(k, on)| (k, on, "3D EPI"))).collect(),
                ReadoutKind::Spiral => grase_only.iter().map(|&(k, on)| (k, on, "GRASE"))
                    .chain(epi3d_only.iter().map(|&(k, on)| (k, on, "3D EPI"))).collect(),
                ReadoutKind::Epi3d => spiral_only.iter().map(|&(k, on)| (k, on, "spiral"))
                    .chain(echo_train_only.iter().map(|&(k, on)| (k, on, "spin-echo train"))).collect(),
            };
            if let Some((key, _, other)) = wrong.iter().find(|(_, on, _)| *on) {
                return Err(format!("overlay: readout.{key} is a {other} key; the readout is {}", kind.as_str()));
            }
            if let Some(v) = opt_num(sidecar, "ParallelReductionFactorOutOfPlane")? {
                if v != 1.0 {
                    return Err(format!("asl.json: ParallelReductionFactorOutOfPlane {v}: out-of-plane acceleration \
                                        (CAIPI) is deferred"));
                }
            }
            if let Some(d) = sidecar.get("PartialFourierDirection").and_then(Value::as_str) {
                if matches!(d.to_ascii_lowercase().as_str(), "k" | "k-" | "slice" | "partition") {
                    return Err(format!("asl.json: PartialFourierDirection {d:?}: partial Fourier along kz is deferred"));
                }
            }
            let pos = |v: Option<f64>, what: &str| -> Result<Option<f64>, String> {
                v.map(|x| require_finite_positive(x, what)).transpose()
            };
            let side_shots = match sidecar.get("NumberShots") {
                None | Some(Value::Null) => None,
                Some(Value::Number(x)) if x.as_f64().is_some_and(|v| v >= 1.0 && v.fract() == 0.0) =>
                    Some(x.as_f64().unwrap() as usize),
                Some(other) => return Err(format!(
                    "asl.json: NumberShots {other}: a positive integer is supported (the before/after-centre array \
                     form is not)")),
            };
            let kz_segments = match r.kz_segments {
                Some(0) => return Err("overlay: readout.kz_segments must be at least 1".to_string()),
                Some(v) => (v, Source::Overlay),
                None => (1, Source::Default),
            };
            // spirals (part C): interleaves and the readout duration have no defensible default;
            // the dwell time is the overlay's, else the sidecar's DwellTime
            let side_dwell = pos(opt_num(sidecar, "DwellTime")?, "asl.json: DwellTime")?;
            let (interleaves, spiral_readout_ms, dwell_time_s) = match kind {
                ReadoutKind::Grase | ReadoutKind::Epi3d => (None, None, side_dwell.map(|d| (d, Source::Sidecar))),
                ReadoutKind::Spiral => {
                    let il = match r.interleaves {
                        Some(0) => return Err("overlay: readout.interleaves must be at least 1".to_string()),
                        Some(v) => v,
                        None => return Err("a spiral readout needs [readout] interleaves (no default: product spirals \
                                            differ between vendors and releases)".to_string()),
                    };
                    let t = pos(r.spiral_readout_time, "overlay readout.spiral_readout_time")?.ok_or(
                        "a spiral readout needs [readout] spiral_readout_time (ms; no default: product spirals differ \
                         between vendors and releases)".to_string())?;
                    let dwell = match (pos(r.dwell_time, "overlay readout.dwell_time")?, side_dwell) {
                        (Some(v), _) => (v, Source::Overlay),
                        (None, Some(v)) => (v, Source::Sidecar),
                        (None, None) => return Err("a spiral readout needs its dwell time: the sidecar has no DwellTime; \
                                                    give [readout] dwell_time (s)".to_string()),
                    };
                    (Some((il, Source::Overlay)), Some((t, Source::Overlay)), Some(dwell))
                }
            };
            // the spiral's radial margin over Nyquist (P5 part C, amended): at exactly Nyquist some
            // image directions are not recoverable
            let radial_oversampling = match (kind, r.radial_oversampling) {
                (ReadoutKind::Grase | ReadoutKind::Epi3d, _) => None,
                (ReadoutKind::Spiral, Some(v)) if v.is_finite() && v >= 1.0 => Some((v, Source::Overlay)),
                (ReadoutKind::Spiral, Some(v)) => {
                    return Err(format!("overlay: readout.radial_oversampling {v} must be at least 1"))
                }
                (ReadoutKind::Spiral, None) => Some((DEFAULT_RADIAL_OVERSAMPLING, Source::Default)),
            };
            // NumberShots: absent means 1 for GRASE; a spiral's interleaves are shots by
            // construction, so absent it is interleaves x kz_segments, and given it must agree
            let number_shots = match (kind, side_shots) {
                (ReadoutKind::Spiral, s) => {
                    let n = interleaves.map_or(1, |v| v.0) * kz_segments.0;
                    if let Some(s) = s {
                        if s != n {
                            return Err(format!("asl.json: NumberShots {s}, but a spiral of {} interleaves x {} kz segments \
                                                has {n} shots", n / kz_segments.0, kz_segments.0));
                        }
                    }
                    (n, if s.is_some() { Source::Sidecar } else { Source::Default })
                }
                (_, Some(s)) => (s, Source::Sidecar),
                (_, None) => (1usize, Source::Default),
            };
            let ky_segments = match r.ky_segments {
                Some(0) => return Err("overlay: readout.ky_segments must be at least 1".to_string()),
                Some(v) => (v, Source::Overlay),
                None if kind == ReadoutKind::Spiral => (1, Source::Default),
                None if number_shots.0 % kz_segments.0 == 0 => (number_shots.0 / kz_segments.0, Source::Default),
                None => return Err(format!(
                    "NumberShots {} does not divide into {} kz segments", number_shots.0, kz_segments.0)),
            };
            if kind != ReadoutKind::Spiral && ky_segments.0 * kz_segments.0 != number_shots.0 {
                return Err(format!(
                    "[readout] ky_segments {} x kz_segments {} = {} shots, but NumberShots is {}",
                    ky_segments.0, kz_segments.0, ky_segments.0 * kz_segments.0, number_shots.0));
            }
            let kz_order = match r.kz_order.as_deref() {
                None => (KzOrder::Centric, Source::Default),
                Some("centric") => (KzOrder::Centric, Source::Overlay),
                Some("linear") => (KzOrder::Linear, Source::Overlay),
                Some(other) => return Err(format!("overlay: readout.kz_order {other:?}: expected \"centric\" or \"linear\"")),
            };
            let refocusing_time_ms = match pos(r.refocusing_time, "overlay readout.refocusing_time")? {
                Some(v) => (v, Source::Overlay),
                None => (2.0, Source::Default),
            };
            let refocusing_flip_deg = match (r.refocusing_flip_angle, side_fa) {
                (Some(v), _) => (v, Source::Overlay),
                (None, Some(v)) => (v, Source::Sidecar),
                (None, None) => (180.0, Source::Default),
            };
            if kind != ReadoutKind::Epi3d && !(refocusing_flip_deg.0 > 0.0 && refocusing_flip_deg.0 <= 180.0) {
                return Err(format!(
                    "refocusing flip angle {} (the sidecar's FlipAngle is read as the refocusing angle of a 3D echo \
                     train; [readout] refocusing_flip_angle overrides it) must be in (0, 180]", refocusing_flip_deg.0));
            }
            let readout_samples = match r.readout_samples {
                Some(0) => return Err("overlay: readout.readout_samples must be at least 1".to_string()),
                v => v,
            };
            Some(ReadoutSpec {
                kind: (kind, kind_src),
                number_shots,
                ky_segments,
                kz_segments,
                kz_order,
                echo_spacing_ms: pos(r.echo_spacing, "overlay readout.echo_spacing")?,
                refocusing_time_ms,
                refocusing_flip_deg,
                line_spacing_ms: pos(r.line_spacing, "overlay readout.line_spacing")?,
                readout_samples,
                effective_echo_spacing_s: pos(opt_num(sidecar, "EffectiveEchoSpacing")?, "asl.json: EffectiveEchoSpacing")?,
                total_readout_time_s: (total_readout_time_s > 0.0).then_some(total_readout_time_s),
                dwell_time_s,
                interleaves,
                spiral_readout_ms,
                radial_oversampling,
                ge3d: None,
            })
        }
    };

    // Background suppression (P3 addendum, part A).
    let bo = overlay.and_then(|o| o.background_suppression.as_ref());
    let suppression = if background_suppression {
        let n_pulses = match field(sidecar, "BackgroundSuppressionNumberPulses")? {
            Value::Number(x) if x.as_f64().is_some_and(|v| v >= 0.0 && v.fract() == 0.0) => x.as_f64().unwrap() as usize,
            other => {
                return Err(format!(
                    "asl.json: BackgroundSuppressionNumberPulses must be a non-negative integer, got {other}"))
            }
        };
        let times: Vec<f64> = match field(sidecar, "BackgroundSuppressionPulseTime")? {
            Value::Array(a) => a
                .iter()
                .map(|x| x.as_f64().ok_or_else(|| "asl.json: BackgroundSuppressionPulseTime holds a non-number".to_string()))
                .collect::<Result<_, _>>()?,
            _ => {
                return Err("asl.json: BackgroundSuppressionPulseTime must be an array of seconds from labeling \
                            start".to_string())
            }
        };
        if times.len() != n_pulses {
            return Err(format!(
                "asl.json: BackgroundSuppressionNumberPulses is {n_pulses} but BackgroundSuppressionPulseTime has \
                 {} entries", times.len()));
        }
        for t in &times {
            require_finite_nonneg(*t, "asl.json: BackgroundSuppressionPulseTime entry")?;
        }
        let epsilon = match bo.and_then(|b| b.inversion_efficiency) {
            Some(v) => {
                if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
                    return Err(format!("overlay: background_suppression.inversion_efficiency must be in [0, 1], got {v}"));
                }
                (v, Source::Overlay)
            }
            None => (0.95, Source::Default),
        };
        let presaturation = match bo.and_then(|b| b.presaturation) {
            Some(v) => (v, Source::Overlay),
            None => (false, Source::Default),
        };
        // Distinct PLDs of the labeled rows, ascending: BIDS defines the pulse times for the
        // first PLD only, and the overlay can supply one array per PLD.
        let mut distinct: Vec<u64> = (0..n).filter(|&i| kinds[i] != RowKind::M0scan).map(|i| pld[i].to_bits()).collect();
        distinct.sort_by(|a, b| f64::from_bits(*a).partial_cmp(&f64::from_bits(*b)).unwrap());
        distinct.dedup();
        let first_pld_pulses = match bo.and_then(|b| b.pulse_times_per_pld.as_ref()) {
            Some(sets) => sets.first().cloned().unwrap_or_default(),
            None => times.clone(),
        };
        let (per_row, first_pld_applied_to_all): (Vec<Vec<f64>>, bool) = match bo.and_then(|b| b.pulse_times_per_pld.as_ref()) {
            Some(sets) => {
                if sets.len() != distinct.len() {
                    return Err(format!(
                        "overlay: background_suppression.pulse_times_per_pld has {} arrays but the series has {} \
                         distinct PostLabelingDelay values", sets.len(), distinct.len()));
                }
                for s in sets {
                    for t in s {
                        require_finite_nonneg(*t, "overlay background_suppression.pulse_times_per_pld entry")?;
                    }
                }
                let per_row = (0..n)
                    .map(|i| {
                        if kinds[i] == RowKind::M0scan {
                            Vec::new()
                        } else {
                            sets[distinct.iter().position(|b| *b == pld[i].to_bits()).unwrap()].clone()
                        }
                    })
                    .collect();
                (per_row, false)
            }
            None => (
                (0..n).map(|i| if kinds[i] == RowKind::M0scan { Vec::new() } else { times.clone() }).collect(),
                distinct.len() > 1,
            ),
        };
        // The model (P4, part D): global-bolus unless asked, and its region keys only with
        // bolus-position, so no key is read for nothing.
        let model_name = bo.and_then(|b| b.model.as_deref()).unwrap_or("global-bolus");
        let model = match model_name {
            "global-bolus" => {
                if bo.is_some_and(|b| b.pulse_region.is_some() || b.slab_entry_time.is_some()) {
                    return Err("overlay: background_suppression.pulse_region and slab_entry_time are read only with \
                                model = \"bolus-position\"; under \"global-bolus\" they would have no effect"
                        .to_string());
                }
                SuppressionModel::GlobalBolus
            }
            "bolus-position" => {
                let region = match (bo.and_then(|b| b.pulse_region.as_deref()), bo.and_then(|b| b.slab_entry_time.as_ref())) {
                    (None, _) => {
                        return Err("overlay: background_suppression.model = \"bolus-position\" needs pulse_region \
                                    (\"global\" or \"slab\")".to_string())
                    }
                    (Some("global"), None) => Region::Global,
                    (Some("global"), Some(_)) => {
                        return Err("overlay: background_suppression.slab_entry_time is read only with \
                                    pulse_region = \"slab\"".to_string())
                    }
                    (Some("slab"), None) => {
                        return Err("overlay: background_suppression.pulse_region = \"slab\" needs slab_entry_time \
                                    (seconds after labeling, or \"arrival\")".to_string())
                    }
                    (Some("slab"), Some(SlabEntry::Seconds(d))) => {
                        Region::Slab(require_finite_nonneg(*d, "overlay background_suppression.slab_entry_time")?)
                    }
                    (Some("slab"), Some(SlabEntry::Word(w))) if w == "arrival" => Region::Arrival,
                    (Some("slab"), Some(SlabEntry::Word(w))) => {
                        return Err(format!(
                            "overlay: background_suppression.slab_entry_time {w:?}: expected seconds or \"arrival\""))
                    }
                    (Some(other), _) => {
                        return Err(format!(
                            "overlay: background_suppression.pulse_region {other:?}: expected \"global\" or \"slab\""))
                    }
                };
                SuppressionModel::BolusPosition(region)
            }
            other => {
                return Err(format!(
                    "overlay: background_suppression.model {other:?}: expected \"global-bolus\" or \"bolus-position\""))
            }
        };
        // P6 part A: a Hadamard row is a decoded sub-bolus, whose PostLabelingDelay runs from its own
        // end, while the pulses run from the start of the whole encoded labeling; hadamard_spec
        // checks them against the raw volumes' timing instead
        let hadamard_rows = overlay.is_some_and(|o| o.hadamard.is_some());
        for (i, r) in rows.iter().enumerate().filter(|_| !hadamard_rows) {
            for &p in &per_row[i] {
                if p >= r.t {
                    return Err(format!(
                        "row {i}: background-suppression pulse at {p} s is at or after the first slice's readout at \
                         {} s; P3 models pulses before the first excitation only (a later pulse would act on the \
                         next repetition of the earlier slices)", r.t));
                }
                // Partial-bolus inversion (P4, part D): a slab pulse never touches blood upstream
                // of the slab, and a PASL bolus is labeled whole at 0, so both may fall inside the
                // bolus; a global pulse during (P)CASL labeling also inverts blood not yet labeled.
                let allowed_early = match model {
                    SuppressionModel::GlobalBolus => false,
                    SuppressionModel::BolusPosition(Region::Global) => label_type == LabelType::Pasl,
                    SuppressionModel::BolusPosition(_) => true,
                };
                if p < r.tau && !allowed_early {
                    let why = match model {
                        SuppressionModel::GlobalBolus => "a pulse during labeling inverts part of the bolus, which the \
                                                          global-bolus model cannot express; use model = \
                                                          \"bolus-position\"",
                        _ => "a global pulse during (P)CASL labeling also inverts blood not yet labeled, whose \
                              recovery until its labeling (the inflowing blood's history) is not modeled; use \
                              pulse_region = \"slab\" if the pulse spares the labeling plane",
                    };
                    return Err(format!(
                        "row {i}: background-suppression pulse at {p} s falls before the bolus end at {} s; {why}",
                        r.tau));
                }
            }
        }
        Some(SuppressionSpec { epsilon, presaturation, per_row, first_pld_pulses, first_pld_applied_to_all, model })
    } else {
        if bo.is_some_and(|b| b.model.is_some() || b.pulse_region.is_some() || b.slab_entry_time.is_some()) {
            return Err("overlay: background_suppression.model, pulse_region and slab_entry_time need \
                        BackgroundSuppression true".to_string());
        }
        None
    };

    // P6 part A: under [hadamard] the motion indexes the raw volumes, H per cycle of H - 1 deltam
    // rows (an inconsistent count is hadamard_spec's error, below)
    let n_motion = match overlay.and_then(|o| o.hadamard.as_ref()).and_then(|h| h.order) {
        Some(order) if order > 1 => {
            let m0 = kinds.iter().filter(|k| **k == RowKind::M0scan).count();
            let d = kinds.iter().filter(|k| **k == RowKind::Deltam).count();
            if d % (order - 1) == 0 { m0 + d / (order - 1) * order } else { n }
        }
        _ => n,
    };
    let motion = overlay_motion(overlay.and_then(|o| o.motion.as_ref()), n_motion, mb, readout.as_ref().map(|r| r.number_shots.0))?;

    // ---- P4 (addendum; plan Task 4): activation and refusal, in the plan's order ----
    // 1. The arterial compartment: on with [macrovascular] or either map; then each quantity
    //    needs exactly one source.
    let mo = overlay.and_then(|o| o.macrovascular.as_ref());
    let (has_abv, has_aatt) = phantom.map_or((false, false), |p| (p.has_abv, p.has_aatt));
    let macro_on = mo.is_some() || has_abv || has_aatt;
    let check_table = |t: &BTreeMap<String, f64>, what: &str, upper: Option<f64>| -> Result<(), String> {
        for (name, v) in t {
            if !(v.is_finite() && *v >= 0.0 && upper.is_none_or(|u| *v <= u)) {
                return Err(format!("overlay: macrovascular.{what}.{name} = {v} is out of range"));
            }
        }
        Ok(())
    };
    let source = |map: bool, table: Option<&BTreeMap<String, f64>>, what: &str, upper: Option<f64>|
        -> Result<QuantitySource, String>
    {
        match (map, table) {
            (true, Some(_)) => Err(format!(
                "{what} is given both by the phantom's map and by overlay [macrovascular]; one source each")),
            (true, None) => Ok(QuantitySource::Map),
            (false, Some(t)) => {
                check_table(t, what, upper)?;
                Ok(QuantitySource::Table(t.clone()))
            }
            (false, None) => Err(format!(
                "the arterial compartment is on but {what} has no source: give the phantom map or an overlay \
                 [macrovascular] {what} table")),
        }
    };
    let macro_partial = if macro_on {
        let abv = source(has_abv, mo.and_then(|m| m.arterial_blood_volume.as_ref()), "arterial_blood_volume", Some(1.0))?;
        let aatt = source(has_aatt, mo.and_then(|m| m.arterial_transit_time.as_ref()), "arterial_transit_time", None)?;
        Some((abv, aatt))
    } else {
        None
    };
    // 2. The arterial T2: only with the compartment; default the resolved blood T2.
    let t2_arterial_in = so.and_then(|s| s.t2_arterial);
    let macrovascular = match macro_partial {
        None => {
            if t2_arterial_in.is_some() {
                return Err("overlay: signal.t2_arterial without an arterial compartment ([macrovascular] or the \
                            abv/aatt maps)".to_string());
            }
            None
        }
        Some((abv, aatt)) => {
            let t2_arterial = match t2_arterial_in {
                Some(v) => (require_finite_positive(v, "overlay signal.t2_arterial")?, Source::Overlay),
                None => (t2_blood_s.0, Source::T2Blood),
            };
            Some(MacroSpec { abv, aatt, t2_arterial })
        }
    };
    // 3. Crushing.
    let co = overlay.and_then(|o| o.vascular_crushing.as_ref());
    let crushing = if opt_bool(sidecar, "VascularCrushing")? {
        let venc = per_row(
            num_or_array(sidecar, "VascularCrushingVENC")
                .map_err(|e| format!("{e} (required with VascularCrushing true)"))?,
            "VascularCrushingVENC", &kinds, false)?;
        for (i, v) in venc.iter().enumerate() {
            if !(*v == 0.0 || (v.is_finite() && *v >= 0.1)) {
                return Err(format!(
                    "asl.json: VascularCrushingVENC = {v} for row {i}: 0 (off) or at least 0.1 cm/s, below which \
                     capillary flow could no longer be assumed unaffected"));
            }
        }
        let no_arterial = co.and_then(|c| c.no_arterial_compartment).unwrap_or(false);
        let velocity = co.and_then(|c| c.arterial_velocity.clone());
        if macrovascular.is_some() {
            if no_arterial {
                return Err("overlay: vascular_crushing.no_arterial_compartment = true contradicts the arterial \
                            compartment that is on".to_string());
            }
            let Some(v) = &velocity else {
                return Err("VascularCrushing true with an arterial compartment needs overlay [vascular_crushing] \
                            arterial_velocity (cm/s per label)".to_string());
            };
            for (name, x) in v {
                // 10 m/s is far above any arterial speed, and keeps v_max / VENC within the
                // range the sine integral is evaluated on (crushing::survival)
                if !(x.is_finite() && (0.0..=MAX_ARTERIAL_VELOCITY).contains(x)) {
                    return Err(format!("overlay: vascular_crushing.arterial_velocity.{name} = {x} is out of range \
                                        (0 to {MAX_ARTERIAL_VELOCITY} cm/s)"));
                }
            }
        } else {
            if !no_arterial {
                return Err("asl.json: VascularCrushing true, but there is no arterial compartment for the crushers \
                            to act on: give [macrovascular], or set [vascular_crushing] no_arterial_compartment = \
                            true to accept that they act on nothing modeled".to_string());
            }
            if velocity.is_some() {
                return Err("overlay: vascular_crushing.arterial_velocity with no arterial compartment would act on \
                            nothing".to_string());
            }
        }
        Some(CrushSpec { venc, arterial_velocity: velocity, no_arterial_compartment: no_arterial })
    } else {
        if sidecar.get("VascularCrushingVENC").is_some_and(|v| !v.is_null()) || co.is_some() {
            return Err("VascularCrushingVENC or [vascular_crushing] without VascularCrushing true".to_string());
        }
        None
    };
    // Part A.
    let exchange_time = match ko.and_then(|k| k.exchange_time) {
        // below a microsecond 1/exchange_time can overflow, and T1'' collapse to a guarded
        // zero that would read as the slow-exchange limit
        Some(v) if v.is_finite() && v >= MIN_EXCHANGE_TIME => Some(v),
        Some(v) => return Err(format!("overlay: kinetic.exchange_time = {v} must be at least {MIN_EXCHANGE_TIME} s")),
        None => None,
    };
    // 7. Physiological noise.
    let physio = match overlay.and_then(|o| o.physio.as_ref()) {
        None => None,
        Some(po) => {
            let d = PhysioParams::default();
            let amp = |v: Option<f64>, what: &str| -> Result<f64, String> {
                let v = v.unwrap_or(0.0);
                if v.is_finite() { Ok(v) } else { Err(format!("overlay: physio.{what} is not finite")) }
            };
            let params = PhysioParams {
                tissue: [amp(po.tissue_cardiac, "tissue_cardiac")?, amp(po.tissue_respiratory, "tissue_respiratory")?,
                         amp(po.tissue_drift, "tissue_drift")?],
                label: [amp(po.label_cardiac, "label_cardiac")?, amp(po.label_respiratory, "label_respiratory")?,
                        amp(po.label_drift, "label_drift")?],
                cardiac_frequency: require_finite_positive(po.cardiac_frequency.unwrap_or(d.cardiac_frequency), "overlay physio.cardiac_frequency")?,
                cardiac_cv: po.cardiac_cv.unwrap_or(d.cardiac_cv),
                respiratory_frequency: require_finite_positive(po.respiratory_frequency.unwrap_or(d.respiratory_frequency), "overlay physio.respiratory_frequency")?,
                respiratory_cv: po.respiratory_cv.unwrap_or(d.respiratory_cv),
                drift_time: require_finite_positive(po.drift_time.unwrap_or(d.drift_time), "overlay physio.drift_time")?,
            };
            let (lo, hi) = PHYSIO_FREQUENCY_RANGE;
            for (what, f) in [("cardiac_frequency", params.cardiac_frequency), ("respiratory_frequency", params.respiratory_frequency)] {
                if !(lo..=hi).contains(&f) {
                    return Err(format!("overlay: physio.{what} = {f} must be in [{lo}, {hi}] Hz"));
                }
            }
            for (what, cv) in [("cardiac_cv", params.cardiac_cv), ("respiratory_cv", params.respiratory_cv)] {
                if !(cv.is_finite() && (0.0..=0.3).contains(&cv)) {
                    return Err(format!("overlay: physio.{what} = {cv} must be in [0, 0.3] (so every period is positive)"));
                }
            }
            // The periodic terms are bounded by their amplitudes; at |a_c| + |a_r| >= 1 a factor
            // could reach zero or flip the sign of the magnetization it scales. (The drift is
            // unbounded in principle; its amplitude is the user's, recorded in the sidecar.)
            for (what, a) in [("tissue", params.tissue), ("label", params.label)] {
                if a[0].abs() + a[1].abs() >= 1.0 {
                    return Err(format!(
                        "overlay: physio {what}_cardiac and {what}_respiratory sum to {} in magnitude; at 1 or more the \
                         {what} factor can reach zero or change sign", a[0].abs() + a[1].abs()));
                }
            }
            if params.tissue.iter().chain(&params.label).all(|a| *a == 0.0) {
                return Err("overlay: [physio] with all six amplitudes zero would modulate nothing; set an amplitude \
                            or remove the table".to_string());
            }
            Some(params)
        }
    };
    let mut row_start = Vec::with_capacity(n);
    let mut clock = 0.0;
    // a segmented 3D volume takes NumberShots repetitions, each with its own labeling (P5 part D)
    let shots = readout.as_ref().map_or(1, |r| r.number_shots.0);
    for r in &rows {
        row_start.push(clock);
        if shots == 1 {
            clock += r.tr;
        } else {
            clock += shots as f64 * r.tr;
        }
    }

    let m0_repetition_time_s = overlay.and_then(|o| o.m0.as_ref()).and_then(|m| m.repetition_time);
    if let Some(v) = m0_repetition_time_s {
        require_finite_positive(v, "overlay m0.repetition_time")?;
        if max_offset > v {
            return Err(format!(
                "overlay m0.repetition_time {v} s is shorter than the last slice offset {max_offset} s of the \
                 readout the M0 scan shares"));
        }
    }
    if m0_type == M0Type::Separate && m0_repetition_time_s.is_none() {
        return Err("M0Type is \"Separate\" but the overlay has no [m0] repetition_time; the ASL sidecar's \
                    RepetitionTimePreparation describes the ASL series, not the M0 scan".to_string());
    }
    let mut acq = overlay_acq(overlay.and_then(|o| o.acquisition.as_ref()))?;
    let seed = overlay.and_then(|o| o.seed).unwrap_or(0);
    if is_3d {
        // the eddy model needs a prep gradient ASL has not, and an echo-dependent eddy evolution
        // would break the 3D path's z factorization (P5 part B)
        for (key, v) in [("eddy_strength", acq.eddy_strength), ("eddy_quad", acq.eddy_quad), ("eddy_phase", acq.eddy_phase)] {
            if v != 0.0 {
                return Err(format!("overlay: acquisition.{key} = {v} with MRAcquisitionType \"3D\": the eddy model \
                                    is not available in 3D"));
            }
        }
    }
    if readout_kind.is_some_and(|k| k.0 == ReadoutKind::Spiral) {
        // GRAPPA, partial Fourier, Nyquist ghosting and spikes are Cartesian phase-encode effects
        // with no spiral meaning in this model (P5 part C)
        for (key, on) in [("partial_fourier", acq.partial_fourier != 1.0), ("ghost_offset", acq.ghost_offset != 0.0),
                          ("n_spikes", acq.n_spikes != 0)] {
            if on {
                return Err(format!("overlay: acquisition.{key} with a spiral readout: a Cartesian phase-encode effect \
                                    with no spiral meaning in this model"));
            }
        }
        if accel != 1 {
            return Err(format!("asl.json: ParallelReductionFactorInPlane {accel} with a spiral readout: GRAPPA has no \
                                spiral meaning in this model"));
        }
    }

    // Compat (P2 addendum, part A): pinned values checked against what was set explicitly.
    let co = overlay.and_then(|o| o.compat.as_ref());
    let asldro = co.and_then(|c| c.asldro).unwrap_or(false);
    let grid_origin = match co.and_then(|c| c.grid_origin.as_deref()) {
        Some(s) => GridOrigin::parse(s).map_err(|e| format!("overlay: compat.{e}"))?,
        None if asldro => GridOrigin::VoxelCentre,
        None => GridOrigin::Corner,
    };
    let compat = if asldro {
        if is_3d {
            return Err("[compat] asldro = true with MRAcquisitionType \"3D\": simasl has no readout to compare a \
                        3D echo train against".to_string());
        }
        let ao = overlay.and_then(|o| o.acquisition.as_ref());
        let explicit: [(&str, Option<f64>); 10] = [
            ("oversample", ao.and_then(|a| a.oversample).map(|v| v as f64)),
            ("partial_fourier", ao.and_then(|a| a.partial_fourier)),
            ("n_coils", ao.and_then(|a| a.n_coils).map(|v| v as f64)),
            ("ghost_offset", ao.and_then(|a| a.ghost_offset)),
            ("n_spikes", ao.and_then(|a| a.n_spikes).map(|v| v as f64)),
            ("eddy_strength", ao.and_then(|a| a.eddy_strength)),
            ("eddy_quad", ao.and_then(|a| a.eddy_quad)),
            ("eddy_phase", ao.and_then(|a| a.eddy_phase)),
            ("signal_scale", ao.and_then(|a| a.signal_scale)),
            ("noise_variance", ao.and_then(|a| a.noise_variance)),
        ];
        for (key, set) in explicit {
            let pinned = COMPAT_PINNED.iter().find(|(k, _)| *k == key).unwrap().1;
            if let Some(v) = set {
                if v != pinned {
                    return Err(format!(
                        "overlay: acquisition.{key} = {v} under [compat] asldro = true, which pins it to {pinned}; \
                         compat mode overrides nothing silently"));
                }
            }
        }
        if let Some(w) = ao.and_then(|a| a.window.as_deref()) {
            if parse_window(w)? != KspaceWindow::None {
                return Err(format!(
                    "overlay: acquisition.window = {w:?} under [compat] asldro = true, which pins it to \"none\""));
            }
        }
        let refuse = |what: String| -> Result<(), String> {
            Err(format!("{what} under [compat] asldro = true: simasl cannot express it"))
        };
        if accel != 1 {
            refuse(format!("asl.json: ParallelReductionFactorInPlane = {accel}, pinned to 1,"))?;
        }
        // The sidecar's own PartialFourier describes the acquisition too; outside compat the
        // overlay's value is what is simulated and the input is kept as replaced, but compat
        // overrides nothing silently.
        if let Some(pf) = opt_num(sidecar, "PartialFourier")? {
            if pf != 1.0 {
                refuse(format!("asl.json: PartialFourier = {pf}, pinned to 1,"))?;
            }
        }
        if mb != 1 {
            refuse(format!("asl.json: MultibandAccelerationFactor = {mb}"))?;
        }
        if background_suppression {
            refuse("asl.json: BackgroundSuppression = true".to_string())?;
        }
        if motion.as_ref().is_some_and(|m| m.within.is_some()) {
            refuse("overlay: [motion.within_volume]".to_string())?;
        }
        if m0_type == M0Type::Separate {
            refuse("asl.json: M0Type \"Separate\" (simasl's M0 is an m0scan row: use M0Type \"Included\")".to_string())?;
        }
        for (part, on) in [
            ("[kinetic] exchange_time (P4 part A)", exchange_time.is_some()),
            ("the arterial compartment (P4 part B)", macrovascular.is_some()),
            ("VascularCrushing (P4 part C)", crushing.is_some()),
            ("background_suppression.model = \"bolus-position\" (P4 part D)",
             suppression.as_ref().is_some_and(|s| s.model != SuppressionModel::GlobalBolus)),
            ("[physio] (P4 part E)", physio.is_some()),
        ] {
            if on {
                refuse(part.to_string())?;
            }
        }
        if slice_offsets.iter().any(|t| *t != 0.0) {
            refuse(format!(
                "asl.json: SliceTiming {timing:?} (unequal entries give each slice its own kinetic time)"))?;
        }
        acq.oversample = 1;
        acq.partial_fourier = 1.0;
        acq.n_coils = 1;
        acq.ghost_offset = 0.0;
        acq.n_spikes = 0;
        acq.eddy_strength = 0.0;
        acq.eddy_quad = 0.0;
        acq.eddy_phase = 0.0;
        acq.signal_scale = 1.0;
        acq.noise_variance = 0.0;
        acq.window = KspaceWindow::None;
        let desired_snr = match co.and_then(|c| c.desired_snr) {
            Some(v) => {
                require_finite_nonneg(v, "overlay compat.desired_snr")?;
                (v > 0.0).then_some(v)
            }
            None => None,
        };
        Some(CompatSpec { desired_snr })
    } else {
        if co.and_then(|c| c.desired_snr).is_some() {
            return Err("overlay: compat.desired_snr is read only with [compat] asldro = true; without it the noise \
                        is acquisition.noise_variance".to_string());
        }
        None
    };

    // P7 part B: under Look-Locker the encoded preparations are read readouts_per_cycle times
    let hadamard_ll = match (look_locker, overlay.and_then(|o| o.hadamard.as_ref())) {
        (true, Some(_)) => match overlay.and_then(|o| o.look_locker.as_ref()).and_then(|l| l.readouts_per_cycle) {
            Some(m) => Some((m, ll_flips.as_deref())),
            None => return Err("[hadamard] with LookLocker: true needs [look_locker] readouts_per_cycle: the readouts \
                                after each encoded preparation (the context lists the decoded volumes, so the cycles \
                                cannot be read off the delays)".to_string()),
        },
        _ => None,
    };
    let hadamard = hadamard_spec(
        overlay.and_then(|o| o.hadamard.as_ref()), label_type, &rows, &pld, suppression.as_ref(), crushing.as_ref(),
        compat.is_some(), hadamard_ll,
    )?;
    let look_locker = look_locker_spec(LlInputs {
        on: look_locker, overlay, rows: &rows, pld: &pld, flips: ll_flips, contrast, ge: ge.as_ref(), is_3d,
        epi3d: readout_kind.is_some_and(|k| k.0 == ReadoutKind::Epi3d),
        hadamard: hadamard.as_ref(), compat: compat.is_some(), suppression: suppression.as_ref(),
        m0_type,
    })?;

    let mo = overlay.and_then(|o| o.multi_te.as_ref());
    let multi_te = if !multi_echo {
        if mo.is_some() {
            return Err("overlay: [multi_te] is read only with more than one echo (one --asl-json per echo)".to_string());
        }
        None
    } else {
        let refocusing_time_ms = match (contrast == Contrast::GradientEcho, mo.and_then(|m| m.refocusing_time)) {
            (true, Some(_)) => {
                return Err("overlay: multi_te.refocusing_time reserves time for spin-echo refocusing pulses, and a \
                            gradient-echo readout has none".to_string())
            }
            (true, None) => None,
            (false, Some(v)) => Some((require_finite_positive(v, "overlay multi_te.refocusing_time")?, Source::Overlay)),
            (false, None) => Some((2.0, Source::Default)),
        };
        let max_image_memory_gib = match (compat.is_some(), mo.and_then(|m| m.max_image_memory_gib)) {
            (false, Some(_)) => {
                return Err("overlay: multi_te.max_image_memory_gib bounds compat's per-echo images; without [compat] \
                            asldro = true every echo shares one image set".to_string())
            }
            (false, None) => None,
            (true, Some(_)) if overlay.and_then(|o| o.images.as_ref()).and_then(|i| i.max_memory_gib).is_some() => {
                return Err("overlay: [images] max_memory_gib and [multi_te] max_image_memory_gib are the same bound \
                            (the second its older name): give one".to_string())
            }
            (true, Some(v)) => Some((require_finite_positive(v, "overlay multi_te.max_image_memory_gib")?, Source::Overlay)),
            // P7: [images] max_memory_gib is the newer name of the same bound
            (true, None) => match overlay.and_then(|o| o.images.as_ref()).and_then(|i| i.max_memory_gib) {
                Some(v) => Some((require_finite_positive(v, "overlay images.max_memory_gib")?, Source::Overlay)),
                None => Some((4.0, Source::Default)),
            },
        };
        Some(MultiTeSpec { sidecars: sidecars.to_vec(), refocusing_time_ms, max_image_memory_gib })
    };

    // P7 part C: the 3D EPI train's own inputs; its slab entry from [readout] or bolus-position's
    let images_gib = overlay.and_then(|o| o.images.as_ref()).and_then(|i| i.max_memory_gib)
        .map(|v| require_finite_positive(v, "overlay images.max_memory_gib")).transpose()?;
    let mut readout = readout;
    if let Some(r) = readout.as_mut().filter(|r| r.kind.0 == ReadoutKind::Epi3d) {
        let empty = ReadoutOverlay::default();
        let ro = overlay.and_then(|o| o.readout.as_ref()).unwrap_or(&empty);
        let excitation_spacing_ms = match ro.excitation_spacing {
            Some(v) => require_finite_positive(v, "overlay readout.excitation_spacing")?,
            None => return Err("[readout] type \"epi3d\" needs excitation_spacing (ms between the train's excitations; \
                                no default: it sets every excitation's longitudinal state)".to_string()),
        };
        let excitation_time_ms = match ro.excitation_time {
            Some(v) => (require_finite_positive(v, "overlay readout.excitation_time")?, Source::Overlay),
            None => (2.0, Source::Default),
        };
        let from_bolus = match suppression.as_ref().map(|s| s.model) {
            Some(SuppressionModel::BolusPosition(Region::Slab(d))) => Some(SlabEntryTime::Seconds(d)),
            Some(SuppressionModel::BolusPosition(Region::Arrival)) => Some(SlabEntryTime::Arrival),
            _ => None,
        };
        let from_readout = match &ro.slab_entry_time {
            None => None,
            Some(SlabEntry::Seconds(d)) => Some(SlabEntryTime::Seconds(require_finite_nonneg(*d, "overlay readout.slab_entry_time")?)),
            Some(SlabEntry::Word(w)) if w == "arrival" => Some(SlabEntryTime::Arrival),
            Some(SlabEntry::Word(w)) => return Err(format!(
                "overlay: readout.slab_entry_time {w:?}: expected seconds or \"arrival\"")),
        };
        let slab_entry = match (from_readout, from_bolus) {
            (Some(_), Some(_)) => return Err(
                "overlay: readout.slab_entry_time and background_suppression.slab_entry_time both give the slab \
                 entry: one source (P4's rule)".to_string()),
            (Some(e), None) => (e, "overlay readout.slab_entry_time"),
            (None, Some(e)) => (e, "overlay background_suppression.slab_entry_time"),
            (None, None) => return Err(
                "[readout] type \"epi3d\" needs the label's slab entry: a 3D excitation covers the slab and depletes \
                 the label in its arteries (P7 part C); give [readout] slab_entry_time (seconds after labeling, or \
                 \"arrival\" for no depletion before the voxel)".to_string()),
        };
        let max_t1_groups = match ro.max_t1_groups {
            Some(0) => return Err("overlay: readout.max_t1_groups must be at least 1".to_string()),
            Some(v) => (v, Source::Overlay),
            None => (16, Source::Default),
        };
        let node_tolerance = match ro.node_tolerance {
            Some(v) if v.is_finite() && v > 0.0 && v < 1.0 => (v, Source::Overlay),
            Some(v) => return Err(format!("overlay: readout.node_tolerance {v} must be in (0, 1)")),
            None => (1e-4, Source::Default),
        };
        if compat.is_some() {
            return Err("[readout] type \"epi3d\" under [compat] asldro = true: simasl has no 3D gradient-echo train"
                .to_string());
        }
        if acq.partial_fourier != 1.0 {
            return Err("overlay: acquisition.partial_fourier with [readout] type \"epi3d\": partial Fourier is not \
                        available on the 3D gradient-echo train".to_string());
        }
        let ll_array = look_locker.as_ref().is_some_and(|l| l.flip_array);
        if !ll_array && ge.as_ref().is_some_and(|g| g.flip == Source::Default) {
            return Err("[readout] type \"epi3d\" requires FlipAngle (the excitation of every partition); without it \
                        the gradient-echo default of 90 degrees would saturate the train".to_string());
        }
        if let Some(g) = ge.as_ref().filter(|g| !(g.flip_deg > 0.0 && g.flip_deg <= 90.0)) {
            return Err(format!(
                "FlipAngle {} with [readout] type \"epi3d\": the train's excitations are in (0, 90] degrees (under \
                 \"epi3d\" FlipAngle is the excitation, not a GRASE refocusing angle)", g.flip_deg));
        }
        let max_memory_gib = match images_gib {
            Some(v) => (v, Source::Overlay),
            None => (4.0, Source::Default),
        };
        r.ge3d = Some(Ge3dSpec { excitation_spacing_ms, excitation_time_ms, slab_entry, max_t1_groups, node_tolerance,
                                 max_memory_gib });
    } else if images_gib.is_some() && !(multi_te.is_some() && compat.is_some()) {
        return Err("overlay: [images] max_memory_gib bounds the images of a 3D gradient-echo series ([readout] type \
                    \"epi3d\") or of compat's multi-TE echoes; this protocol has neither".to_string());
    }

    Ok(Protocol {
        echo_times_s, multi_te, hadamard, look_locker,
        compat, grid_origin, exchange_time, macrovascular, crushing, physio, row_start,
        label_type, rows, m0_type, background_suppression, suppression, ir, ge, readout, motion, mb_interleaved,
        slice_offsets, field_strength, voxel_size_mm, reverse_phase, phase_encoding_direction: ped,
        echo_time_s, total_readout_time_s, accel, mb, alpha, lambda, t1b, t2_blood_s, contrast,
        m0_repetition_time_s, seed, acq, input_sidecar: sidecar.clone(),
    })
}

/// Read the files and [`parse`]. A relative `motion.trajectory` path is taken relative to the
/// overlay file's directory.
pub fn load(asl_json: &Path, aslcontext_tsv: &Path, overlay: Option<&Path>, phantom: Option<&PhantomParams>)
    -> Result<Protocol, String>
{
    load_with(&[asl_json], aslcontext_tsv, overlay, phantom, false)
}

/// [`load`], with `compat_asldro` the CLI's `--compat-asldro`: the same as writing
/// `[compat] asldro = true` in the overlay (an overlay that says `false` is an error). `asl_json`
/// is one sidecar, or one per echo in echo order (P6 addendum, part C).
pub fn load_with(
    asl_json: &[&Path], aslcontext_tsv: &Path, overlay: Option<&Path>, phantom: Option<&PhantomParams>, compat_asldro: bool,
) -> Result<Protocol, String> {
    let mut sidecars = Vec::with_capacity(asl_json.len());
    for path in asl_json {
        let sidecar: Value = serde_json::from_str(
            &std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        sidecars.push(sidecar);
    }
    let ctx = std::fs::read_to_string(aslcontext_tsv).map_err(|e| format!("{}: {e}", aslcontext_tsv.display()))?;
    let mut ov: Option<Overlay> = match overlay {
        None => None,
        Some(p) => Some(
            toml::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?)
                .map_err(|e| format!("{}: {e}", p.display()))?,
        ),
    };
    if let (Some(op), Some(o)) = (overlay, ov.as_mut()) {
        if let Some(t) = o.motion.as_mut().and_then(|m| m.trajectory.as_mut()) {
            let p = Path::new(t.as_str());
            if p.is_relative() {
                if let Some(dir) = op.parent() {
                    *t = dir.join(p).to_string_lossy().to_string();
                }
            }
        }
    }
    if compat_asldro {
        let c = ov.get_or_insert_with(Overlay::default).compat.get_or_insert_with(CompatOverlay::default);
        if c.asldro == Some(false) {
            return Err("--compat-asldro with an overlay that sets [compat] asldro = false".to_string());
        }
        c.asldro = Some(true);
    }
    parse_echoes(&sidecars, &ctx, ov.as_ref(), phantom)
}

impl Protocol {
    /// Whether any P6 feature is on (Hadamard encoding, Look-Locker readouts, more than one
    /// echo). Decided once, here: false sends the series down today's code unchanged (the legacy
    /// dispatch). Each P6 part adds its condition as it is parsed.
    pub fn p6_active(&self) -> bool {
        self.echo_times_s.len() > 1 || self.hadamard.is_some() || self.look_locker.is_some() || self.ge3d().is_some()
    }

    /// The 3D gradient-echo train's inputs, when the readout is one (P7 part C).
    pub fn ge3d(&self) -> Option<&Ge3dSpec> {
        self.readout.as_ref().and_then(|r| r.ge3d.as_ref())
    }

    /// The kinetic constants for `row`.
    pub fn kinetic(&self, row: &Row) -> Kinetic {
        Kinetic { label_type: self.label_type, tau: row.tau, alpha: self.alpha.0, lambda: self.lambda.0, t1b: self.t1b.0 }
    }

    /// Blood T2 in milliseconds for the acquisition stage. One of the two conversion sites in this
    /// module (with [`Protocol::acquisition`]); nothing downstream converts again.
    pub fn t2_blood_ms(&self) -> f32 {
        (self.t2_blood_s.0 * 1000.0) as f32
    }

    /// `TotalAcquiredPairs` as BIDS defines it: control-label pairs, or the count of
    /// pre-subtracted `deltam` volumes when there are no pairs.
    pub fn total_acquired_pairs(&self) -> usize {
        let labels = self.rows.iter().filter(|r| r.kind == RowKind::Label).count();
        if labels > 0 { labels } else { self.rows.iter().filter(|r| r.kind == RowKind::Deltam).count() }
    }

    /// The acquisition stage's parameters for an acquired matrix `[nx, ny]`, in milliseconds,
    /// with the readout-timing check run here so its failure names `EchoTime` and
    /// `TotalReadoutTime`. `do_distortions` is set by the caller from whether a fieldmap exists.
    pub fn acquisition(&self, nx: usize, ny: usize) -> Result<Acquisition, String> {
        let a = &self.acq;
        let acq = Acquisition {
            // TotalReadoutTime * 1000 / ny: the inverse of trxscan.rs:968, ny not ny - 1.
            t_line: self.total_readout_time_s * 1000.0 / ny as f64,
            t_echo: self.echo_time_s * 1000.0,
            t_inhom: a.t_inhom_ms,
            signal_scale: a.signal_scale,
            reverse_phase: self.reverse_phase,
            do_distortions: true,
            // Under compat the readout applies no relaxation: simasl's one exp(-TE/T2) per voxel
            // is applied in the signal stage instead (P2 addendum, part A).
            do_relaxation: self.compat.is_none(),
            noise_variance: a.noise_variance,
            partial_fourier: a.partial_fourier,
            pf_mode: a.pf_mode,
            ghost_offset: a.ghost_offset,
            eddy_strength: a.eddy_strength,
            eddy_quad: a.eddy_quad,
            eddy_phase: a.eddy_phase,
            eddy_tau: a.eddy_tau_ms,
            n_spikes: a.n_spikes,
            spike_amplitude: a.spike_amplitude,
            window: a.window,
            n_coils: a.n_coils,
            accel: self.accel,
            acs_lines: a.acs_lines,
            seed: self.seed,
            echo: if self.contrast == Contrast::GradientEcho { EchoFormation::Gradient } else { EchoFormation::Spin },
        };
        // a 3D echo train's timing is resolve_readout's (its lines are timed from each echo)
        if self.readout.is_none() {
            mrsim_acq::kspace::validate_acquisition_timing(&acq, nx, ny).map_err(|inner| {
                format!(
                    "EchoTime {} s with TotalReadoutTime {} s on a {ny}-line readout: the first acquired line \
                     precedes the excitation ({inner}). Raise EchoTime or bring TotalReadoutTime below about \
                     {} s; partial Fourier and in-plane acceleration do not shorten the pre-echo readout in \
                     this model.",
                    self.echo_time_s, self.total_readout_time_s, 2.0 * self.echo_time_s)
            })?;
        }
        Ok(acq)
    }
}

/// Half the sampled block of one 2D EPI readout (s) on an `[nx, ny]` acquired matrix: the
/// quantity `mrsim_acq`'s `SingleShotEpi::half_read_time` computes (`kx_max * ky_max * dt / 2`,
/// `dt = t_line / kx_max`), from the `t_line` of [`Protocol::acquisition`], in the same order.
pub fn half_epi_block_s(p: &Protocol, nx: usize, ny: usize) -> f64 {
    let t_line = p.total_readout_time_s * 1000.0 / ny as f64;
    let dt = t_line / nx as f64;
    (nx * ny) as f64 * dt / 2.0 / 1000.0
}

/// The excitation timing of a multi-TE protocol on its acquired grid (P6 addendum, part C,
/// "Timing on explicit intervals"), in seconds. Within one excitation at `exc`, echo `e` samples
/// `[exc + TE_e - h, exc + TE_e + h]`: under gradient echo the blocks must not overlap; under spin
/// echo the first refocusing pulse is centred at `TE_1 / 2` and each later one at
/// `(TE_e + TE_e+1) / 2`, each reserving `refocusing_time` about its centre, which must clear the
/// excitation and the blocks on either side. Between excitations, slices sharing an offset are one
/// excitation group (multiband groups): a group's last block must end before the next group's
/// excitation, and the last group's
/// before the repetition time, for every row (`row.t + offset` for an ASL row, the offset alone
/// for an m0scan row) and for the separate M0 at its own repetition time. Under compat every
/// slice offset is zero (compat refuses unequal `SliceTiming`: simasl's slices are independent and
/// excited together), so there is one group and no between-group check: compat's abstraction,
/// recorded in the sidecar; the checks within the excitation and against TR still run.
pub fn check_excitation_timing(p: &Protocol, acq_dims: [usize; 3]) -> Result<(), String> {
    let tes = &p.echo_times_s;
    let h = half_epi_block_s(p, acq_dims[0], acq_dims[1]);
    if let Some(ll) = &p.look_locker {
        return check_look_locker_timing(p, ll, h);
    }
    let block = |e: usize| [tes[e] - h, tes[e] + h];
    if block(0)[0] < 0.0 {
        return Err(format!(
            "echo 1 at {} s samples from {} s before its excitation (half readout {h} s)", tes[0], -block(0)[0]));
    }
    let refocusing = p.multi_te.as_ref().and_then(|m| m.refocusing_time_ms).map(|r| r.0 / 1000.0);
    match refocusing {
        None => {
            for e in 1..tes.len() {
                if block(e)[0] < block(e - 1)[1] {
                    return Err(format!(
                        "gradient echo: echo {} samples [{}, {}] s after the excitation and echo {} [{}, {}] s; the \
                         readouts overlap (TotalReadoutTime {} s)", e, block(e - 1)[0], block(e - 1)[1], e + 1,
                        block(e)[0], block(e)[1], p.total_readout_time_s));
                }
            }
        }
        Some(r) => {
            let c = tes[0] / 2.0;
            if c - r / 2.0 < 0.0 || c + r / 2.0 > block(0)[0] {
                return Err(format!(
                    "spin echo: the first refocusing pulse, centred at {c} s with {r} s reserved (multi_te.refocusing_time), \
                     must start after the excitation and end before echo 1's readout starts at {} s", block(0)[0]));
            }
            for e in 1..tes.len() {
                let c = (tes[e - 1] + tes[e]) / 2.0;
                if c - r / 2.0 < block(e - 1)[1] || c + r / 2.0 > block(e)[0] {
                    return Err(format!(
                        "spin echo: the refocusing pulse between echoes {} and {}, centred at {c} s with {r} s reserved \
                         (multi_te.refocusing_time), must fit between echo {}'s readout ending at {} s and echo {}'s \
                         starting at {} s", e, e + 1, e, block(e - 1)[1], e + 1, block(e)[0]));
                }
            }
        }
    }
    let last_end = block(tes.len() - 1)[1];
    let mut groups: Vec<f64> = p.slice_offsets.clone();
    groups.sort_by(f64::total_cmp);
    groups.dedup();
    for w in groups.windows(2) {
        if w[0] + last_end > w[1] {
            return Err(format!(
                "the slices excited at {} s read their last echo until {} s, after the next slices' excitation at {} s \
                 (SliceTiming)", w[0], w[0] + last_end, w[1]));
        }
    }
    let g_last = groups.last().copied().unwrap_or(0.0);
    for (i, r) in p.rows.iter().enumerate() {
        let exc = if r.kind == RowKind::M0scan { 0.0 } else { r.t };
        if exc + g_last + last_end > r.tr {
            return Err(format!(
                "row {i}: the last slice's last echo ends at {} s, after its RepetitionTimePreparation {} s",
                exc + g_last + last_end, r.tr));
        }
    }
    if let (M0Type::Separate, Some(tr)) = (p.m0_type, p.m0_repetition_time_s) {
        if g_last + last_end > tr {
            return Err(format!(
                "the separate M0's last slice's last echo ends at {} s, after its repetition time {tr} s (overlay \
                 m0.repetition_time)", g_last + last_end));
        }
    }
    Ok(())
}

/// Complete a 3D protocol's readout with the acquisition grid (P5 plan, Task 7): the segment
/// divisions, the line spacing (P5 addendum, "Timing from BIDS": the overlay, else BIDS's
/// effective spacing times the ky segments, else the dwell time as a recorded lower bound), the
/// echo spacing from `EchoTime` read as the k-space-centre time, and every timing check on the
/// actual RF and sampling intervals, for every row and for the separate M0. `None` for 2D.
/// `series` calls it after computing the grid and the CLI before writing anything.
/// The 3D gradient-echo train completed with the acquisition grid (P7 addendum, part C).
#[derive(Debug, Clone, PartialEq)]
pub struct Ge3dResolution {
    pub train: ExcitationTrain,
    pub readout: Ge3dReadout,
    pub n_shots: usize,
    /// Lines per EPI block and excitations per shot.
    pub epi: usize,
    pub n_exc: usize,
    /// The excitation (0-based, within its shot) that reads the kz centre.
    pub e_c: usize,
    pub t_line_ms: f64,
    pub t_line_source: &'static str,
    /// The centre line's time from its echo (ms): BIDS's EchoTime is the k-space centre, so each
    /// echo's block is centred at `EchoTime - t_kyc`.
    pub t_kyc_ms: f64,
}

/// The `"epi3d"` readout with the grid (P7 addendum, part C): the line spacing (the overlay's, else
/// BIDS's effective spacing times the ky segments, the two agreeing within 1% when both are given),
/// the echoes' blocks centred so the k-space centre line is read at each `EchoTime`, and the timing
/// of every row's train (from its excitation at `row.t`, an m0scan row's at the start of its
/// repetition) and of the separate M0's. `None` unless the readout is one.
pub fn resolve_ge3d(p: &Protocol, acq_dims: [usize; 3]) -> Result<Option<Ge3dResolution>, String> {
    let (Some(r), Some(g)) = (&p.readout, p.ge3d()) else { return Ok(None) };
    let [_, ny, nz] = acq_dims;
    let ky_segments = r.ky_segments.0;
    let ees = r.effective_echo_spacing_s;
    let trt_ees = r.total_readout_time_s.filter(|_| ny > 1).map(|t| t / (ny as f64 - 1.0));
    if let (Some(a), Some(b)) = (ees, trt_ees) {
        if (a - b).abs() > 0.01 * a {
            return Err(format!(
                "EffectiveEchoSpacing {a} s and TotalReadoutTime {} s disagree: BIDS defines TotalReadoutTime = \
                 EffectiveEchoSpacing (ny - 1) = {} s on {ny} lines", r.total_readout_time_s.unwrap(), a * (ny as f64 - 1.0)));
        }
    }
    let effective = ees.or(trt_ees);
    let (t_line_ms, t_line_source) = match (r.line_spacing_ms, effective, r.dwell_time_s) {
        (Some(v), eff, _) => {
            if let Some(e) = eff {
                let want = e * 1000.0 * ky_segments as f64;
                if (v - want).abs() > 0.01 * want {
                    return Err(format!(
                        "[readout] line_spacing {v} ms disagrees with the sidecar's effective spacing {e} s x {ky_segments} \
                         ky segments = {want} ms"));
                }
            }
            (v, "overlay readout.line_spacing")
        }
        (None, Some(e), _) => (e * 1000.0 * ky_segments as f64, "effective echo spacing x ky segments"),
        // as GRASE's (P5 part B)
        (None, None, Some((d, _))) => {
            let samples = r.readout_samples.unwrap_or(acq_dims[0]);
            (samples as f64 * d * 1000.0, "DwellTime x readout samples (a lower bound: no ramps, no receiver oversampling)")
        }
        (None, None, None) => return Err(
            "a 3D EPI readout needs its line spacing: the sidecar has no EffectiveEchoSpacing, TotalReadoutTime or \
             DwellTime; give [readout] line_spacing (ms) or effective echo spacing".to_string()),
    };
    let block = grase_block(ny, ky_segments, t_line_ms, p.reverse_phase)?;
    let t_kyc_ms = block.t_ms[ny / 2];
    let train = ExcitationTrain {
        kz_segments: r.kz_segments.0,
        kz_order: r.kz_order.0,
        exc_spacing_ms: g.excitation_spacing_ms,
        echo_times_ms: p.echo_times_s.iter().map(|te| te * 1000.0 - t_kyc_ms).collect(),
        excitation_time_ms: g.excitation_time_ms.0,
    };
    let readout = Ge3dReadout { ky_segments, t_line_ms, reverse_phase: p.reverse_phase };
    let table = ge3d_lines(&train, &readout, ny, nz).map_err(|e| {
        if nz < 2 { format!("{e}: use MRAcquisitionType 2D") } else { e }
    })?;
    for (i, row) in p.rows.iter().enumerate() {
        let exc = if row.kind == RowKind::M0scan { 0.0 } else { row.t };
        check_ge3d_timing(&train, &table, exc * 1000.0, row.tr * 1000.0).map_err(|e| format!("row {i}: {e}"))?;
    }
    if let (M0Type::Separate, Some(tr)) = (p.m0_type, p.m0_repetition_time_s) {
        check_ge3d_timing(&train, &table, 0.0, tr * 1000.0).map_err(|e| format!("the separate M0: {e}"))?;
    }
    // P7 part C, 3D Look-Locker: each readout's sub-train must end before the next one's first
    // excitation pulse
    if let Some(ll) = &p.look_locker {
        let half = table.block.t_ms.iter().fold(0.0f64, |m, t| m.max(t.abs())) + table.block.t_line_ms / 2.0;
        let te_last = train.echo_times_ms[train.echo_times_ms.len() - 1];
        let length = (table.n_exc as f64 - 1.0) * train.exc_spacing_ms + te_last + half;
        for c in ll.cycles.iter().filter(|c| !c.m0scan) {
            for w in c.rows.windows(2) {
                let (a, b) = (p.rows[w[0]].t * 1000.0, p.rows[w[1]].t * 1000.0);
                if a + length > b - train.excitation_time_ms / 2.0 + 1e-9 {
                    return Err(format!(
                        "row {}: its Look-Locker sub-train ({} excitations {} ms apart from {:.4} ms) reads until {:.4} ms, \
                         after the next readout's first excitation pulse at {:.4} ms (row {}): the readouts must be at \
                         least {:.4} ms apart", w[0], table.n_exc, train.exc_spacing_ms, a, a + length,
                        b - train.excitation_time_ms / 2.0, w[1], length + train.excitation_time_ms / 2.0));
                }
            }
        }
    }
    Ok(Some(Ge3dResolution {
        n_shots: table.n_shots, epi: table.block.epi, n_exc: table.n_exc, e_c: table.e_c, train, readout, t_line_ms,
        t_line_source, t_kyc_ms,
    }))
}

pub fn resolve_readout(p: &Protocol, acq_dims: [usize; 3]) -> Result<Option<ReadoutResolution>, String> {
    let Some(r) = &p.readout else { return Ok(None) };
    let [nx, ny, nz] = acq_dims;
    // the echo train's amplitudes need the blood's T1 (its EPG), from whichever source it came
    if !(p.t1b.0.is_finite() && p.t1b.0 > 0.0) {
        return Err(format!(
            "the arterial blood T1 is {} s ({}): a 3D echo train's echo amplitudes need a positive T1", p.t1b.0,
            p.t1b.1.as_str()));
    }
    if r.kind.0 == ReadoutKind::Spiral {
        return resolve_spiral(p, r, acq_dims).map(Some);
    }
    if r.kind.0 == ReadoutKind::Epi3d {
        return Err("a 3D gradient-echo train ([readout] type \"epi3d\") is resolved by resolve_ge3d".to_string());
    }
    let (ky_segments, kz_segments) = (r.ky_segments.0, r.kz_segments.0);
    if ny % ky_segments != 0 {
        let fit = (ny / ky_segments).max(1) * ky_segments;
        return Err(format!(
            "{ny} phase-encode lines do not divide into {ky_segments} ky segments: an [acquisition] matrix with \
             {fit} (or {}) lines would", fit + ky_segments));
    }
    if nz % kz_segments != 0 {
        return Err(format!(
            "{nz} partitions (the phantom's extent at the slab's voxel size) do not divide into {kz_segments} kz \
             segments; crop the phantom or change kz_segments"));
    }
    let (epi, etl) = (ny / ky_segments, nz / kz_segments);
    // BIDS's effective spacing: EffectiveEchoSpacing, else TotalReadoutTime / (ny - 1) (BIDS's
    // definition, the new 3D resolver only); both given must agree
    let ees = r.effective_echo_spacing_s;
    let trt_ees = r.total_readout_time_s.filter(|_| ny > 1).map(|t| t / (ny as f64 - 1.0));
    if let (Some(a), Some(b)) = (ees, trt_ees) {
        if (a - b).abs() > 0.01 * a {
            return Err(format!(
                "EffectiveEchoSpacing {a} s and TotalReadoutTime {} s disagree: BIDS defines TotalReadoutTime = \
                 EffectiveEchoSpacing (ny - 1) = {} s on {ny} lines", r.total_readout_time_s.unwrap(), a * (ny as f64 - 1.0)));
        }
    }
    let effective = ees.or(trt_ees);
    let (t_line_ms, t_line_source) = match (r.line_spacing_ms, effective, r.dwell_time_s) {
        (Some(v), eff, _) => {
            if let Some(e) = eff {
                let want = e * 1000.0 * ky_segments as f64;
                if (v - want).abs() > 0.01 * want {
                    return Err(format!(
                        "[readout] line_spacing {v} ms disagrees with the sidecar's effective spacing {e} s x {ky_segments} \
                         ky segments = {want} ms"));
                }
            }
            (v, "overlay readout.line_spacing")
        }
        (None, Some(e), _) => (e * 1000.0 * ky_segments as f64, "effective echo spacing x ky segments"),
        (None, None, Some((d, _))) => {
            let samples = r.readout_samples.unwrap_or(nx);
            (samples as f64 * d * 1000.0, "DwellTime x readout samples (a lower bound: no ramps, no receiver oversampling)")
        }
        (None, None, None) => return Err(
            "a GRASE readout needs its line spacing: the sidecar has no EffectiveEchoSpacing, TotalReadoutTime or \
             DwellTime; give [readout] line_spacing (ms) or effective echo spacing".to_string()),
    };
    // what is published is what is simulated: an overlay spacing accepted within 1% of the
    // sidecar's replaces it (the sidecar's value goes to InputValuesReplaced)
    let effective = match r.line_spacing_ms {
        Some(_) => effective.map(|_| t_line_ms / 1000.0 / ky_segments as f64),
        None => effective,
    };
    let block = grase_block(ny, ky_segments, t_line_ms, p.reverse_phase)?;
    let probe = EchoTrain { etl, esp_ms: 1.0, refocusing_deg: r.refocusing_flip_deg.0, kz_order: r.kz_order.0, kz_segments,
                            refocusing_time_ms: r.refocusing_time_ms.0 };
    let e_c = centre_echo(&probe, nz)?;
    let t_kyc_ms = block.t_ms[ny / 2];
    let te_ms = p.echo_time_s * 1000.0;
    let esp_ms = match r.echo_spacing_ms {
        Some(esp) => {
            let te_implied = e_c as f64 * esp + t_kyc_ms;
            if (te_implied - te_ms).abs() > 1e-3 {
                return Err(format!(
                    "[readout] echo_spacing {esp} ms puts the k-space centre at {te_implied} ms ({e_c} echo(es) and the \
                     centre line {t_kyc_ms} ms from its echo), but EchoTime is {te_ms} ms"));
            }
            esp
        }
        None => esp_from_echo_time(te_ms, e_c, &block),
    };
    if !(esp_ms.is_finite() && esp_ms > 0.0) {
        return Err(format!("EchoTime {te_ms} ms gives a non-positive echo spacing {esp_ms} ms"));
    }
    let train = EchoTrain { esp_ms, ..probe };
    let readout = Readout3d::Grase { ky_segments, t_line_ms, reverse_phase: p.reverse_phase };
    let table = grase_lines(&train, &readout, ny, nz)?;
    // every distinct (excitation, repetition) of the series, and the separate M0's
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let excitations = p.rows.iter().map(|row| (row.t, row.tr)).chain(p.m0_repetition_time_s.map(|tr| (0.0, tr)));
    for (t_exc, tr) in excitations {
        if seen.contains(&(t_exc.to_bits(), tr.to_bits())) {
            continue;
        }
        seen.push((t_exc.to_bits(), tr.to_bits()));
        check_grase_timing(&train, &table, t_exc * 1000.0, tr * 1000.0)?;
    }
    Ok(Some(ReadoutResolution {
        n_shots: table.n_shots, train, readout, epi, etl, t_line_ms, t_line_source, effective_spacing_s: effective, esp_ms,
        e_c, t_kyc_ms, spiral: None,
    }))
}

/// [`resolve_readout`] for the stack of spirals (P5 addendum, part C): a square matrix, the kz
/// division, `ESP = EchoTime / e_c` (each spiral starts at its echo, so the k-space centre is read
/// at `e_c ESP`), the trajectory with its sampling bound, and the timing checks on the actual
/// intervals for every excitation and the separate M0.
fn resolve_spiral(p: &Protocol, r: &ReadoutSpec, acq_dims: [usize; 3]) -> Result<ReadoutResolution, String> {
    let [nx, ny, nz] = acq_dims;
    if !cfg!(feature = "kspace") {
        return Err("a spiral readout needs aslscan built with the `kspace` feature (the time-segmented NUFFT forward)"
            .to_string());
    }
    if nx != ny {
        return Err(format!("a spiral readout needs a square in-plane matrix, not {nx} x {ny}: set [acquisition] matrix"));
    }
    let kz_segments = r.kz_segments.0;
    if nz % kz_segments != 0 {
        return Err(format!(
            "{nz} partitions (the phantom's extent at the slab's voxel size) do not divide into {kz_segments} kz \
             segments; crop the phantom or change kz_segments"));
    }
    let etl = nz / kz_segments;
    let (Some((interleaves, _)), Some((readout_ms, _)), Some((dwell_s, dwell_source)), Some(radial_oversampling)) =
        (r.interleaves, r.spiral_readout_ms, r.dwell_time_s, r.radial_oversampling) else {
        unreachable!("a parsed spiral carries its interleaves, readout time and dwell time")
    };
    let dwell_ms = dwell_s * 1000.0;
    let probe = EchoTrain { etl, esp_ms: 1.0, refocusing_deg: r.refocusing_flip_deg.0, kz_order: r.kz_order.0, kz_segments,
                            refocusing_time_ms: r.refocusing_time_ms.0 };
    let e_c = centre_echo(&probe, nz)?;
    let te_ms = p.echo_time_s * 1000.0;
    let esp_ms = match r.echo_spacing_ms {
        Some(esp) => {
            if (e_c as f64 * esp - te_ms).abs() > 1e-3 {
                return Err(format!(
                    "[readout] echo_spacing {esp} ms puts the k-space centre (the start of echo {e_c}'s spiral) at {} ms, \
                     but EchoTime is {te_ms} ms", e_c as f64 * esp));
            }
            esp
        }
        None => spiral_esp_from_echo_time(te_ms, e_c),
    };
    let train = EchoTrain { esp_ms, ..probe };
    let readout = Readout3d::Spiral { interleaves, readout_ms, dwell_ms, radial_oversampling: radial_oversampling.0 };
    let table = spiral_lines(&train, &readout, nx, ny, nz)?;
    debug_assert_eq!(table.n_shots, r.number_shots.0);
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let excitations = p.rows.iter().map(|row| (row.t, row.tr)).chain(p.m0_repetition_time_s.map(|tr| (0.0, tr)));
    for (t_exc, tr) in excitations {
        if seen.contains(&(t_exc.to_bits(), tr.to_bits())) {
            continue;
        }
        seen.push((t_exc.to_bits(), tr.to_bits()));
        check_spiral_timing(&train, &table, t_exc * 1000.0, tr * 1000.0)?;
    }
    let tj = &table.traj;
    Ok(ReadoutResolution {
        n_shots: table.n_shots, train, readout, epi: 0, etl, t_line_ms: 0.0, t_line_source: "", effective_spacing_s: None,
        esp_ms, e_c, t_kyc_ms: 0.0,
        spiral: Some(SpiralResolution {
            interleaves, readout_ms, dwell_ms, dwell_source, radial_oversampling, samples_per_interleaf: tj.n_samples(),
            k_max: tj.k_max,
            n_turns: tj.n_turns, tau_c_ms: tj.tau_c_ms,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> Value {
        json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Separate", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [3.5, 3.5, 5],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05, 0.10], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.016
        })
    }
    const CTX: &str = "volume_type\ncontrol\nlabel\ncontrol\nlabel\n";
    fn overlay(s: &str) -> Overlay {
        toml::from_str(s).unwrap()
    }
    fn m0_overlay() -> Overlay {
        overlay("[m0]\nrepetition_time = 8.0\n")
    }
    fn fixture(name: &str) -> (Value, String) {
        let d = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/");
        let j = std::fs::read_to_string(format!("{d}{name}/asl.json")).unwrap();
        let c = std::fs::read_to_string(format!("{d}{name}/aslcontext.tsv")).unwrap();
        (serde_json::from_str(&j).unwrap(), c)
    }

    #[test]
    fn pcasl_signal_time_adds_the_labeling_duration() {
        let p = parse(&base(), CTX, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.rows.len(), 4);
        for r in &p.rows {
            assert!((r.t - 3.6).abs() < 1e-12 && (r.tau - 1.8).abs() < 1e-12 && r.tr == 4.0, "{r:?}");
        }
        assert_eq!(p.rows[1].kind, RowKind::Label);
    }

    #[test]
    fn pasl_signal_time_is_the_pld_itself() {
        let mut s = base();
        s["ArterialSpinLabelingType"] = json!("PASL");
        s["BolusCutOffFlag"] = json!(true);
        s["BolusCutOffTechnique"] = json!("Q2TIPS");
        s["BolusCutOffDelayTime"] = json!(0.7);
        s.as_object_mut().unwrap().remove("LabelingDuration");
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        for r in &p.rows {
            assert!((r.t - 1.8).abs() < 1e-12 && (r.tau - 0.7).abs() < 1e-12, "{r:?}");
        }
        assert_eq!(p.alpha, (0.98, Source::Default));
        // Q2TIPS: two-element array, bolus duration is the first
        s["BolusCutOffDelayTime"] = json!([0.7, 1.6]);
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        assert!((p.rows[0].tau - 0.7).abs() < 1e-12);
        s["BolusCutOffDelayTime"] = json!([1.6, 0.7]);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).is_err());
        // PLD <= cutoff: the cutoff pulse would fall after the readout
        s["BolusCutOffDelayTime"] = json!(1.8);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("cutoff"));
        // no flag: no bolus duration to assume
        s["BolusCutOffDelayTime"] = json!(0.7);
        s["BolusCutOffFlag"] = json!(false);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("BolusCutOffFlag"));
    }

    #[test]
    fn array_timing_expands_per_row_including_m0scan_zeros() {
        let mut s = base();
        s["M0Type"] = json!("Included");
        s["PostLabelingDelay"] = json!([0.0, 1.0, 1.0, 2.0, 2.0]);
        s["LabelingDuration"] = json!([0.0, 1.8, 1.8, 1.8, 1.8]);
        s["RepetitionTimePreparation"] = json!([8.0, 4.0, 4.0, 4.0, 4.0]);
        let ctx = "volume_type\nm0scan\ncontrol\nlabel\ncontrol\nlabel\n";
        let p = parse(&s, ctx, None, None).unwrap();
        assert_eq!(p.rows[0], Row { kind: RowKind::M0scan, t: 0.0, tau: 0.0, tr: 8.0 });
        assert!((p.rows[1].t - 2.8).abs() < 1e-12 && (p.rows[3].t - 3.8).abs() < 1e-12);
        assert_eq!(p.rows[4].tr, 4.0);
        // wrong length names both counts
        s["PostLabelingDelay"] = json!([1.0, 1.0, 2.0, 2.0]);
        let e = parse(&s, ctx, None, None).unwrap_err();
        assert!(e.contains("4 entries") && e.contains("5 rows"), "{e}");
        // nonzero m0scan entry
        s["PostLabelingDelay"] = json!([0.5, 1.0, 1.0, 2.0, 2.0]);
        assert!(parse(&s, ctx, None, None).unwrap_err().contains("m0scan"));
        // scalar with an m0scan row: the m0scan row still gets zero timing
        s["PostLabelingDelay"] = json!(1.5);
        s["LabelingDuration"] = json!(1.8);
        let p = parse(&s, ctx, None, None).unwrap();
        assert_eq!(p.rows[0].t, 0.0);
        assert!((p.rows[1].t - 3.3).abs() < 1e-12);
        // a ONE-element array is an array, held to the per-volume length rule
        s["PostLabelingDelay"] = json!([1.5]);
        let e = parse(&s, ctx, None, None).unwrap_err();
        assert!(e.contains("1 entries") && e.contains("5 rows"), "{e}");
        s["PostLabelingDelay"] = json!([]);
        assert!(parse(&s, ctx, None, None).unwrap_err().contains("empty"));
    }

    #[test]
    fn invalid_values_are_rejected_before_simulation() {
        let mut s = base();
        s["RepetitionTimePreparation"] = json!(-4.0);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("RepetitionTimePreparation"));
        let mut s = base();
        s["PostLabelingDelay"] = json!(-0.5);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("PostLabelingDelay"));
        let mut s = base();
        s["TotalReadoutTime"] = json!(-0.016);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("TotalReadoutTime"));
        let mut s = base();
        s["EchoTime"] = json!([]);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("empty"));
        let mut s = base();
        s["LookLocker"] = json!(true);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("LookLocker: true needs"));
        let mut s = base();
        s["VascularCrushing"] = json!(true);
        // P4 accepts crushing, but not without its VENC (and, part C, an arterial compartment)
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("VascularCrushingVENC"));
        let mut s = base();
        s.as_object_mut().unwrap().remove("BackgroundSuppression");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("BackgroundSuppression"));
        let mut s = base();
        s["M0Type"] = json!("Estimate");
        assert!(parse(&s, CTX, None, None).unwrap_err().contains("M0Estimate"));
        s["M0Estimate"] = json!(100.0);
        assert!(parse(&s, CTX, None, None).is_ok());
        // PASL: technique required, at most two cutoff times
        let mut s = base();
        s["ArterialSpinLabelingType"] = json!("PASL");
        s["BolusCutOffFlag"] = json!(true);
        s["BolusCutOffDelayTime"] = json!(0.7);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("BolusCutOffTechnique"));
        s["BolusCutOffTechnique"] = json!("Q2TIPS");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).is_ok());
        s["BolusCutOffDelayTime"] = json!([0.7, 1.0, 1.6]);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("3 entries"));
        // multiband: every excitation must hold exactly mb slices
        let mut s = base();
        s["MultibandAccelerationFactor"] = json!(2);
        s["SliceTiming"] = json!([0.0, 0.0, 0.0, 0.05]);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("shared by 3"));
        // overlay ranges
        let ov = overlay("[acquisition]\npartial_fourier = 1.5\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap_err().contains("partial_fourier"));
        let ov = overlay("[signal]\nt2_blood = -0.1\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap_err().contains("t2_blood"));
        let ov = overlay("[m0]\nrepetition_time = 0.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap_err().contains("m0.repetition_time"));
        assert_eq!(parse(&base(), CTX, Some(&m0_overlay()), None).unwrap().total_acquired_pairs(), 2);
    }

    #[test]
    fn m0type_and_rows_must_agree() {
        let mut s = base();
        // Separate with an m0scan row
        let e = parse(&s, "volume_type\nm0scan\ncontrol\nlabel\n", Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("Separate") && e.contains("row 0"), "{e}");
        // Included without one
        s["M0Type"] = json!("Included");
        assert!(parse(&s, CTX, None, None).unwrap_err().contains("no m0scan"));
        // Separate without the overlay TR
        s["M0Type"] = json!("Separate");
        assert!(parse(&s, CTX, None, None).unwrap_err().contains("repetition_time"));
        // cbf rows are rejected
        s["M0Type"] = json!("Absent");
        assert!(parse(&s, "volume_type\ncbf\n", None, None).unwrap_err().contains("cbf"));
    }

    #[test]
    fn readout_geometry_rules() {
        // 3D with 2D slice timing is refused (BIDS: SliceTiming must not be defined for 3D)
        let mut s = base();
        s["MRAcquisitionType"] = json!("3D");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("SliceTiming"));
        s["MRAcquisitionType"] = json!("1D");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("2D or 3D"));
        let mut s = base();
        s.as_object_mut().unwrap().remove("SliceTiming");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("SliceTiming"));
        let mut s = base();
        s["SliceEncodingDirection"] = json!("k-");
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.slice_offsets, vec![0.10, 0.05, 0.0]);
        s["SliceEncodingDirection"] = json!("i");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).is_err());
        let mut s = base();
        s["PhaseEncodingDirection"] = json!("j");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap().reverse_phase);
        s["PhaseEncodingDirection"] = json!("j-");
        assert!(!parse(&s, CTX, Some(&m0_overlay()), None).unwrap().reverse_phase);
        s["PhaseEncodingDirection"] = json!("i");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("second data axis"));
        let mut s = base();
        s["EchoTime"] = json!([0.012, 0.012]);
        assert!((parse(&s, CTX, Some(&m0_overlay()), None).unwrap().echo_time_s - 0.012).abs() < 1e-15);
        s["EchoTime"] = json!([0.012, 0.030]);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("echo-N"));
        let mut s = base();
        s["MultibandAccelerationFactor"] = json!(3);
        s["SliceTiming"] = json!([0.0, 0.05, 0.0, 0.05, 0.0, 0.05]);
        assert_eq!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap().mb, 3);
        s["SliceTiming"] = json!([0.0, 0.05, 0.10, 0.0, 0.05, 0.10]);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("Multiband"));
    }

    #[test]
    fn acquisition_derivation_and_the_trf_check() {
        let p = parse(&base(), CTX, Some(&m0_overlay()), None).unwrap();
        let a = p.acquisition(64, 64).unwrap();
        assert!((a.t_echo - 12.0).abs() < 1e-12 && (a.t_line - 0.25).abs() < 1e-12);
        assert!(!a.reverse_phase && a.accel == 1 && a.acs_lines == 24 && a.signal_scale == 100.0);
        // The spec's example: TE 12 ms, 64 lines at 0.5 ms/line (TRT 32 ms) starts before excitation.
        let mut s = base();
        s["TotalReadoutTime"] = json!(0.032);
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        let e = p.acquisition(64, 64).unwrap_err();
        assert!(e.contains("EchoTime") && e.contains("TotalReadoutTime") && e.contains("partial Fourier"), "{e}");
    }

    #[test]
    fn defaults_and_precedence_are_recorded() {
        let p = parse(&base(), CTX, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.alpha, (0.85, Source::Default));
        assert_eq!(p.lambda, (0.9, Source::Default));
        assert_eq!(p.t1b, (1.65, Source::Default));
        assert_eq!(p.t2_blood_s, (0.165, Source::Default));
        assert_eq!(p.seed, 0);
        let ph = PhantomParams { lambda: Some(0.91), t1b: Some(1.7), field_strength: Some(3.0), ..Default::default() };
        let p = parse(&base(), CTX, Some(&m0_overlay()), Some(&ph)).unwrap();
        assert_eq!(p.lambda, (0.91, Source::Phantom));
        assert_eq!(p.t1b, (1.7, Source::Phantom));
        let mut s = base();
        s["LabelingEfficiency"] = json!(0.88);
        let p = parse(&s, CTX, Some(&m0_overlay()), Some(&ph)).unwrap();
        assert_eq!(p.alpha, (0.88, Source::Sidecar));
        let ov = overlay("seed = 7\n[kinetic]\nlabel_efficiency = 0.8\nt1_arterial_blood = 1.6\n[signal]\nt2_blood = 0.15\n[m0]\nrepetition_time = 8.0\n[acquisition]\noversample = 4\nwindow = \"fermi:0.45,0.05\"\npf_mode = \"contiguous\"\nnoise_variance = 2.0\n");
        let p = parse(&s, CTX, Some(&ov), Some(&ph)).unwrap();
        assert_eq!(p.alpha, (0.8, Source::Overlay));
        assert_eq!(p.t1b, (1.6, Source::Overlay));
        assert_eq!(p.lambda, (0.91, Source::Phantom));
        assert_eq!(p.t2_blood_s, (0.15, Source::Overlay));
        assert_eq!(p.seed, 7);
        assert_eq!(p.acq.oversample, 4);
        assert_eq!(p.acq.window, KspaceWindow::Fermi { radius: 0.45, width: 0.05 });
        assert_eq!(p.acq.pf_mode, PartialFourierMode::Contiguous);
        assert_eq!(p.acq.noise_variance, 2.0);
        // 1.5 T defaults, and an unknown field strength needs the overlay
        let mut s = base();
        s["MagneticFieldStrength"] = json!(1.5);
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.t1b, (1.35, Source::Default));
        assert_eq!(p.t2_blood_s, (0.290, Source::Default));
        s["MagneticFieldStrength"] = json!(7);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("overlay"));
        // field-strength mismatch with the phantom
        let ph15 = PhantomParams { field_strength: Some(1.5), ..Default::default() };
        assert!(parse(&base(), CTX, Some(&m0_overlay()), Some(&ph15)).unwrap_err().contains("field"));
        // ge parses through the overlay (P5 part A; its own test below)
        let ov = overlay("[signal]\nacq_contrast = \"ge\"\n[m0]\nrepetition_time = 8.0\n");
        assert_eq!(parse(&base(), CTX, Some(&ov), None).unwrap().contrast, Contrast::GradientEcho);
        // unknown overlay keys are refused rather than ignored
        assert!(toml::from_str::<Overlay>("[kinetic]\nlabel_eficiency = 0.8\n").is_err());
    }

    #[test]
    fn real_sidecars_parse_or_fail_for_the_documented_reason() {
        // asl002: Philips 2D EPI PCASL, Separate M0, 70 rows, j, SliceTiming of 20.
        let (s, c) = fixture("asl002");
        let p = parse(&s, &c, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.rows.len(), 70);
        assert!(p.reverse_phase && p.background_suppression);
        assert_eq!(p.slice_offsets.len(), 20);
        assert!((p.rows[0].t - 3.8).abs() < 1e-12);
        let sup = p.suppression.as_ref().unwrap();
        assert_eq!(sup.per_row[0], vec![2.05, 3.276]);
        assert_eq!(sup.epsilon, (0.95, Source::Default));
        assert!(!sup.first_pld_applied_to_all);
        assert!((crate::longitudinal::label_factor(&sup.for_row(0)) - 0.81).abs() < 1e-12);
        // asl004: 2D PCASL with a 96-entry PLD array, LabelingEfficiency in the sidecar, and a
        // trailing blank line in aslcontext.tsv; six PLDs, so BIDS' first-PLD pulse times apply
        // to every row and the sidecar says so.
        let (s, c) = fixture("asl004");
        let p = parse(&s, &c, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.rows.len(), 96);
        assert_eq!(p.alpha, (0.88, Source::Sidecar));
        let sup = p.suppression.as_ref().unwrap();
        assert!(sup.first_pld_applied_to_all);
        assert_eq!(sup.per_row[95], vec![1.428, 1.604]);
        // ...unless the overlay gives one array per PLD (six here; five is refused)
        let ov = overlay("[background_suppression]\npulse_times_per_pld = [[1.42], [1.43], [1.44], [1.45], [1.46], [1.47]]\n[m0]\nrepetition_time = 8.0\n");
        let p = parse(&s, &c, Some(&ov), None).unwrap();
        assert!(!p.suppression.as_ref().unwrap().first_pld_applied_to_all);
        assert_eq!(p.suppression.as_ref().unwrap().per_row[0], vec![1.42]);
        assert_eq!(p.suppression.as_ref().unwrap().per_row[95], vec![1.47]);
        let ov = overlay("[background_suppression]\npulse_times_per_pld = [[1.42], [1.43], [1.44], [1.45], [1.46]]\n[m0]\nrepetition_time = 8.0\n");
        let e = parse(&s, &c, Some(&ov), None).unwrap_err();
        assert!(e.contains("5 arrays") && e.contains("6 distinct"), "{e}");
        assert!((p.rows[0].t - (0.25 + 1.4)).abs() < 1e-12 && (p.rows[95].t - (1.5 + 1.4)).abs() < 1e-12);
        // Its TE 14 ms / TRT 60 ms readout starts before the excitation under this line-timing model.
        let e = p.acquisition(58, 58).unwrap_err();
        assert!(e.contains("EchoTime"), "{e}");
        // The 3D datasets (P5): asl003 as given fails on its PASL delays before the bolus cutoff;
        // asl005 on its missing phase-encode direction (GRASE needs one); asl001 (a spiral) on its
        // missing interleaves, which have no default (asl001_p5 supplies them).
        let (s, c) = fixture("asl003");
        let e = parse(&s, &c, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("cutoff"), "asl003: {e}");
        let (s, c) = fixture("asl005");
        let e = parse(&s, &c, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("phase_encoding_direction"), "asl005: {e}");
        let (s, c) = fixture("asl001");
        let e = parse(&s, &c, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("interleaves"), "asl001: {e}");
        // asl003 as a 2D variant: PASL Q2TIPS with a 20-entry PLD array whose first entries
        // (0.3 s) precede the 0.7 s bolus cutoff, which the PASL check must refuse...
        let (mut s, c) = fixture("asl003");
        s["MRAcquisitionType"] = json!("2D");
        s["SliceTiming"] = json!([0.0, 0.04, 0.08]);
        s["TotalReadoutTime"] = json!(0.02);
        // its 3.5 s TR cannot hold the shifted 4.0 s PLD below; the readout-in-TR check would fire
        s["RepetitionTimePreparation"] = json!(5.0);
        // its pulses (0.15, 0.2 s) precede the 0.7 s bolus cutoff, which the pulse check refuses
        s["BackgroundSuppression"] = json!(false);
        // its 180-degree GRASE refocusing flip angle is not a spin-echo excitation
        s["FlipAngle"] = json!(90);
        let e = parse(&s, &c, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("cutoff") && e.contains("row 0"), "{e}");
        // ...and with every PLD shifted past the cutoff it parses, tau being the first cutoff time.
        let shifted: Vec<f64> = s["PostLabelingDelay"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() + 1.0).collect();
        s["PostLabelingDelay"] = json!(shifted);
        let p = parse(&s, &c, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.label_type, LabelType::Pasl);
        assert_eq!(p.rows.len(), 20);
        assert!((p.rows[0].t - 1.3).abs() < 1e-12 && (p.rows[0].tau - 0.7).abs() < 1e-12);
        assert!((p.rows[19].t - 4.0).abs() < 1e-12);
    }

    /// base(): tau 1.8, PLD 1.8 (t = 3.6), TR 4.0, offsets up to 0.10.
    fn with_suppression(pulses: &[f64]) -> Value {
        let mut s = base();
        s["BackgroundSuppression"] = json!(true);
        s["BackgroundSuppressionNumberPulses"] = json!(pulses.len());
        s["BackgroundSuppressionPulseTime"] = json!(pulses);
        s
    }

    #[test]
    fn background_suppression_inputs_and_checks() {
        let p = parse(&with_suppression(&[2.0, 3.2]), CTX, Some(&m0_overlay()), None).unwrap();
        let sup = p.suppression.as_ref().unwrap();
        assert_eq!(sup.per_row, vec![vec![2.0, 3.2]; 4]);
        assert_eq!(sup.epsilon, (0.95, Source::Default));
        assert_eq!(sup.presaturation, (false, Source::Default));
        assert!(!sup.first_pld_applied_to_all);
        assert!(sup.for_row(0).has_events());
        // zero pulses is allowed (and is the P1 steady state)
        let p = parse(&with_suppression(&[]), CTX, Some(&m0_overlay()), None).unwrap();
        assert!(!p.suppression.as_ref().unwrap().for_row(0).has_events());
        // m0scan rows get no pulses
        let mut s = with_suppression(&[2.0, 3.2]);
        s["M0Type"] = json!("Included");
        s["PostLabelingDelay"] = json!([0.0, 1.8, 1.8]);
        s["LabelingDuration"] = json!([0.0, 1.8, 1.8]);
        s["RepetitionTimePreparation"] = json!([8.0, 4.0, 4.0]);
        let p = parse(&s, "volume_type\nm0scan\ncontrol\nlabel\n", None, None).unwrap();
        assert!(p.suppression.as_ref().unwrap().per_row[0].is_empty());
        assert_eq!(p.suppression.as_ref().unwrap().per_row[1], vec![2.0, 3.2]);
        // required fields, lengths, ranges
        let mut s = with_suppression(&[2.0, 3.2]);
        s.as_object_mut().unwrap().remove("BackgroundSuppressionNumberPulses");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("BackgroundSuppressionNumberPulses"));
        let mut s = with_suppression(&[2.0, 3.2]);
        s["BackgroundSuppressionNumberPulses"] = json!(3);
        let e = parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("is 3") && e.contains("2 entries"), "{e}");
        let mut s = with_suppression(&[2.0, 3.2]);
        s["BackgroundSuppressionPulseTime"] = json!(2.0);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("array"));
        // a pulse at or after the first slice readout (3.6), and one before the bolus end (1.8)
        let e = parse(&with_suppression(&[2.0, 3.6]), CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("3.6 s") && e.contains("first"), "{e}");
        let e = parse(&with_suppression(&[1.5, 3.2]), CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("1.5 s") && e.contains("bolus") && e.contains("bolus-position"), "{e}");
        // overlay values and ranges
        let ov = overlay("[background_suppression]\ninversion_efficiency = 1.0\npresaturation = true\n[m0]\nrepetition_time = 8.0\n");
        let p = parse(&with_suppression(&[2.0, 3.2]), CTX, Some(&ov), None).unwrap();
        assert_eq!(p.suppression.as_ref().unwrap().epsilon, (1.0, Source::Overlay));
        assert_eq!(p.suppression.as_ref().unwrap().presaturation, (true, Source::Overlay));
        let ov = overlay("[background_suppression]\ninversion_efficiency = 1.5\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&with_suppression(&[2.0, 3.2]), CTX, Some(&ov), None).unwrap_err().contains("inversion_efficiency"));
        // the overlay block is ignored when the sidecar says no suppression
        let ov = overlay("[background_suppression]\ninversion_efficiency = 0.5\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap().suppression.is_none());
        // multi-PLD without the override: first PLD's times everywhere, flagged
        let mut s = with_suppression(&[2.0, 3.2]);
        s["PostLabelingDelay"] = json!([1.8, 1.8, 2.0, 2.0]);
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        assert!(p.suppression.as_ref().unwrap().first_pld_applied_to_all);
        let ov = overlay("[background_suppression]\npulse_times_per_pld = [[2.0, 3.2], [2.0, 3.4]]\n[m0]\nrepetition_time = 8.0\n");
        let p = parse(&s, CTX, Some(&ov), None).unwrap();
        assert_eq!(p.suppression.as_ref().unwrap().per_row[3], vec![2.0, 3.4]);
        assert!(!p.suppression.as_ref().unwrap().first_pld_applied_to_all);
        assert_eq!(p.suppression.as_ref().unwrap().first_pld_pulses, vec![2.0, 3.2]);
        // the override's first set is what the standard field will publish
        let ov = overlay("[background_suppression]\npulse_times_per_pld = [[2.1], [2.0, 3.4]]\n[m0]\nrepetition_time = 8.0\n");
        let p = parse(&s, CTX, Some(&ov), None).unwrap();
        assert_eq!(p.suppression.as_ref().unwrap().first_pld_pulses, vec![2.1]);
    }

    #[test]
    fn every_slice_must_read_inside_the_repetition() {
        let mut s = base();
        s["RepetitionTimePreparation"] = json!(3.65); // t 3.6 + last offset 0.10 = 3.7
        let e = parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("3.7") && e.contains("3.65"), "{e}");
        s["RepetitionTimePreparation"] = json!(3.7);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).is_ok());
        // an included m0scan row reads from t = 0 and must still hold the slice offsets...
        let mut s = base();
        s["M0Type"] = json!("Included");
        s["RepetitionTimePreparation"] = json!([0.05, 4.0, 4.0]);
        s["PostLabelingDelay"] = json!([0.0, 1.8, 1.8]);
        s["LabelingDuration"] = json!([0.0, 1.8, 1.8]);
        let e = parse(&s, "volume_type\nm0scan\ncontrol\nlabel\n", None, None).unwrap_err();
        assert!(e.contains("row 0") && e.contains("0.05"), "{e}");
        // ...and so must the separate M0 scan's repetition time
        let ov = overlay("[m0]\nrepetition_time = 0.05\n");
        let e = parse(&base(), CTX, Some(&ov), None).unwrap_err();
        assert!(e.contains("m0.repetition_time") && e.contains("0.1"), "{e}");
    }

    /// P5 part A: gradient echo's excitation angle (overlay over sidecar over 90, signed as IR's),
    /// its refusals, and the echo formation it hands the acquisition stage.
    #[test]
    fn gradient_echo_inputs_and_rules() {
        let ge = |extra: &str| overlay(&format!("[signal]\nacq_contrast = \"ge\"\n{extra}[m0]\nrepetition_time = 8.0\n"));
        let p = parse(&base(), CTX, Some(&ge("")), None).unwrap();
        assert_eq!(p.contrast, Contrast::GradientEcho);
        assert_eq!(p.ge, Some(GeSpec { flip_deg: 90.0, flip: Source::Default }));
        assert!(p.ir.is_none());
        assert_eq!(p.acquisition(64, 64).unwrap().echo, EchoFormation::Gradient);
        let mut s = base();
        s["FlipAngle"] = json!(60);
        assert_eq!(parse(&s, CTX, Some(&ge("")), None).unwrap().ge, Some(GeSpec { flip_deg: 60.0, flip: Source::Sidecar }));
        assert_eq!(parse(&s, CTX, Some(&ge("excitation_flip_angle = 30\n")), None).unwrap().ge,
                   Some(GeSpec { flip_deg: 30.0, flip: Source::Overlay }));
        s["FlipAngle"] = json!(330);
        assert_eq!(parse(&s, CTX, Some(&ge("")), None).unwrap().ge.unwrap().flip_deg, -30.0);
        // no inversion is simulated
        let mut ti = base();
        ti["InversionTime"] = json!(0.5);
        assert!(parse(&ti, CTX, Some(&ge("")), None).unwrap_err().contains("InversionTime"));
        assert!(parse(&base(), CTX, Some(&ge("inversion_time = 0.5\n")), None).unwrap_err().contains("InversionTime"));
        assert!(parse(&base(), CTX, Some(&ge("inversion_flip_angle = 120\n")), None).unwrap_err().contains("inversion_flip_angle"));
        assert!(parse(&base(), CTX, Some(&ge("excitation_flip_angle = 200\n")), None).unwrap_err().contains("[-180, 180]"));
        // the spin echo keeps its own echo formation
        assert_eq!(parse(&base(), CTX, Some(&m0_overlay()), None).unwrap().acquisition(64, 64).unwrap().echo, EchoFormation::Spin);
    }

    #[test]
    fn inversion_recovery_inputs_and_rules() {
        // spin echo refuses a non-90 FlipAngle and any inversion field
        let mut s = base();
        s["FlipAngle"] = json!(90);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap().ir.is_none());
        s["FlipAngle"] = json!(60);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("90-degree"));
        let mut s = base();
        s["InversionTime"] = json!(1.0);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("InversionTime"));
        let ov = overlay("[signal]\nexcitation_flip_angle = 60\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap_err().contains("excitation_flip_angle"));
        // ...but an overlay that states the modeled 90 degrees is fine
        let ov = overlay("[signal]\nexcitation_flip_angle = 90\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).is_ok());
        // BIDS range on the sidecar angle, normalised to simasl's signed convention
        let mut s = base();
        s["FlipAngle"] = json!(400);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("[0, 360]"));
        let ov = overlay("[signal]\ninversion_flip_angle = 120\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap_err().contains("inversion_flip_angle"));
        // ir: defaults, then sidecar, then overlay
        let ir = |extra: &str| overlay(&format!("[signal]\nacq_contrast = \"ir\"\n{extra}[m0]\nrepetition_time = 8.0\n"));
        let p = parse(&base(), CTX, Some(&ir("")), None).unwrap();
        assert_eq!(p.contrast, Contrast::InversionRecovery);
        let spec = p.ir.as_ref().unwrap();
        assert_eq!(spec.params, IrParams { inversion_time: 1.0, excitation_flip_deg: 90.0, inversion_flip_deg: 180.0 });
        assert_eq!((spec.inversion_time, spec.excitation_flip, spec.inversion_flip), (Source::Default, Source::Default, Source::Default));
        let mut s = base();
        s["FlipAngle"] = json!(60);
        s["InversionTime"] = json!(0.5);
        let spec = parse(&s, CTX, Some(&ir("")), None).unwrap().ir.unwrap();
        assert_eq!(spec.params, IrParams { inversion_time: 0.5, excitation_flip_deg: 60.0, inversion_flip_deg: 180.0 });
        assert_eq!((spec.inversion_time, spec.excitation_flip), (Source::Sidecar, Source::Sidecar));
        let spec = parse(&s, CTX, Some(&ir("inversion_time = 0.8\nexcitation_flip_angle = -30\ninversion_flip_angle = 150\n")), None).unwrap().ir.unwrap();
        assert_eq!(spec.params, IrParams { inversion_time: 0.8, excitation_flip_deg: -30.0, inversion_flip_deg: 150.0 });
        assert_eq!((spec.inversion_time, spec.excitation_flip, spec.inversion_flip), (Source::Overlay, Source::Overlay, Source::Overlay));
        // a sidecar FlipAngle of 330 (what the writer emits for -30) reads back as -30
        s["FlipAngle"] = json!(330);
        let spec = parse(&s, CTX, Some(&ir("")), None).unwrap().ir.unwrap();
        assert_eq!(spec.params.excitation_flip_deg, -30.0);
        assert_eq!(spec.excitation_flip, Source::Sidecar);
        // ranges and simasl's TR >= TE + TI
        assert!(parse(&base(), CTX, Some(&ir("excitation_flip_angle = 200\n")), None).unwrap_err().contains("[-180, 180]"));
        assert!(parse(&base(), CTX, Some(&ir("inversion_time = -0.1\n")), None).unwrap_err().contains("inversion time"));
        let e = parse(&base(), CTX, Some(&ir("inversion_time = 3.995\n")), None).unwrap_err();
        assert!(e.contains("EchoTime") && e.contains("inversion"), "{e}");
        // ir with suppression is refused naming both
        let e = parse(&with_suppression(&[2.0, 3.2]), CTX, Some(&ir("")), None).unwrap_err();
        assert!(e.contains("\"ir\"") && e.contains("BackgroundSuppression"), "{e}");
    }

    #[test]
    fn multiband_schedule_must_be_representable() {
        let mut s = base();
        s["MultibandAccelerationFactor"] = json!(2);
        // the counterexample: passes the mb-times check, groups {0, 2} and {1, 3} in the module
        s["SliceTiming"] = json!([0.0, 0.0, 0.05, 0.05]);
        let e = parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("[0, 1]") && e.contains("g + 2"), "{e}");
        // sequential: group g = z % 4 fires in order 0, 1, 2, 3
        s["SliceTiming"] = json!([0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3]);
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        assert!(p.mb == 2 && !p.mb_interleaved);
        // interleaved: groups 0, 2 fire first, then 1, 3
        s["SliceTiming"] = json!([0.0, 0.2, 0.1, 0.3, 0.0, 0.2, 0.1, 0.3]);
        let p = parse(&s, CTX, Some(&m0_overlay()), None).unwrap();
        assert!(p.mb_interleaved);
        // any other order
        s["SliceTiming"] = json!([0.0, 0.3, 0.1, 0.2, 0.0, 0.3, 0.1, 0.2]);
        let e = parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("order [0, 2, 3, 1]"), "{e}");
        // mb 1: no schedule to derive
        assert!(!parse(&base(), CTX, Some(&m0_overlay()), None).unwrap().mb_interleaved);
    }

    #[test]
    fn motion_overlay_resolves_or_is_refused() {
        let mo = |extra: &str| overlay(&format!("[motion]\n{extra}[m0]\nrepetition_time = 8.0\n"));
        assert!(parse(&base(), CTX, Some(&mo("")), None).unwrap().motion.is_none());
        assert!(parse(&base(), CTX, Some(&mo("mode = \"off\"\n")), None).unwrap().motion.is_none());
        let p = parse(&base(), CTX, Some(&mo("mode = \"random\"\ntrans_mm = [1.0, 0.5, 0.0]\nvolumes = [1, 3]\n")), None).unwrap();
        let m = p.motion.as_ref().unwrap();
        assert_eq!(m.mode_name, "random");
        assert!(matches!(&m.mode, MotionMode::Random { trans_mm: [1.0, 0.5, 0.0], rot_deg: [0.0, 0.0, 0.0], volumes } if *volumes == vec![1, 3]));
        assert!(m.within.is_none());
        let p = parse(&base(), CTX, Some(&mo("mode = \"linear\"\nrot_deg = [0.0, 0.0, 2.0]\n")), None).unwrap();
        assert!(matches!(&p.motion.as_ref().unwrap().mode, MotionMode::Linear { volumes, .. } if volumes.len() == 4));
        assert!(parse(&base(), CTX, Some(&mo("mode = \"random\"\n")), None).unwrap_err().contains("moves nothing"));
        assert!(parse(&base(), CTX, Some(&mo("mode = \"random\"\ntrans_mm = [1.0, 0.0, 0.0]\nvolumes = [4]\n")), None).unwrap_err().contains("index 4"));
        assert!(parse(&base(), CTX, Some(&mo("mode = \"linear\"\ntrans_mm = [1.0, 0.0, 0.0]\nvolumes = [1, 1]\n")), None).unwrap_err().contains("more than once"));
        assert!(parse(&base(), CTX, Some(&mo("mode = \"wobble\"\n")), None).unwrap_err().contains("wobble"));
        assert!(parse(&base(), CTX, Some(&mo("mode = \"trajectory\"\n")), None).unwrap_err().contains("motion.trajectory"));
        // trajectory: four rows for four volumes, radians in the file
        let dir = std::env::temp_dir().join(format!("aslscan-motion-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tsv = dir.join("motion.tsv");
        std::fs::write(&tsv, "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n0\t0\t0\t0\t0\t0\n2\t0\t0\t0\t0\t0\n0\t0\t0\t0\t0\t0.0174533\n0\t0\t0\t0\t0\t0\n").unwrap();
        let path = tsv.to_string_lossy().replace('\\', "\\\\");
        let p = parse(&base(), CTX, Some(&mo(&format!("mode = \"trajectory\"\ntrajectory = \"{path}\"\n"))), None).unwrap();
        match &p.motion.as_ref().unwrap().mode {
            MotionMode::Trajectory { poses } => {
                assert_eq!(poses.len(), 4);
                assert_eq!(poses[1].trans_mm, [2.0, 0.0, 0.0]);
                assert!((poses[2].rot_deg[2] - 1.0).abs() < 1e-3);
            }
            other => panic!("{other:?}"),
        }
        let e = parse(&base(), "volume_type\ncontrol\nlabel\n", Some(&mo(&format!("mode = \"trajectory\"\ntrajectory = \"{path}\"\n"))), None).unwrap_err();
        assert!(e.contains("4 rows") && e.contains("2 volumes"), "{e}");
        assert!(parse(&base(), CTX, Some(&mo(&format!("mode = \"random\"\ntrans_mm = [1, 0, 0]\ntrajectory = \"{path}\"\n"))), None).unwrap_err().contains("only read"));
        // corrupt rows are refused before mrsim-acq's lenient loader sees them: a truncated
        // row, a NaN, a word; `n/a` is the documented zero
        let traj = |body: &str| {
            std::fs::write(&tsv, format!("trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n{body}")).unwrap();
            parse(&base(), CTX, Some(&mo(&format!("mode = \"trajectory\"\ntrajectory = \"{path}\"\n"))), None)
        };
        let ok = "0\t0\t0\t0\t0\t0\n";
        let e = traj(&format!("{ok}{ok}{ok}1\t0\t0\n")).unwrap_err();
        assert!(e.contains("row 3") && e.contains("rot_x") && e.contains("missing"), "{e}");
        let e = traj(&format!("{ok}{ok}{ok}0\tNaN\t0\t0\t0\t0\n")).unwrap_err();
        assert!(e.contains("row 3") && e.contains("trans_y"), "{e}");
        assert!(traj(&format!("{ok}{ok}{ok}0\t0\tabc\t0\t0\t0\n")).unwrap_err().contains("trans_z"));
        assert!(traj(&format!("{ok}{ok}{ok}n/a\tn/a\tn/a\tn/a\tn/a\tn/a\n")).is_ok());
        // within-volume events need multiband
        let wv = "[motion.within_volume]\ndropout_rate = 0.2\nseverity = 0.5\n";
        assert!(parse(&base(), CTX, Some(&mo(wv)), None).unwrap_err().contains("MultibandAccelerationFactor"));
        let mut s = base();
        s["MultibandAccelerationFactor"] = json!(2);
        s["SliceTiming"] = json!([0.0, 0.05, 0.0, 0.05]);
        let p = parse(&s, CTX, Some(&mo(wv)), None).unwrap();
        let m = p.motion.as_ref().unwrap();
        assert!(matches!(m.mode, MotionMode::Off) && m.mode_name == "off");
        assert_eq!(m.within.as_ref().unwrap(), &WithinVolume { dropout_rate: 0.2, severity: 0.5, jump_mm: [0.0; 3], jump_deg: [0.0; 3] });
        assert!(parse(&s, CTX, Some(&mo("[motion.within_volume]\ndropout_rate = 1.2\nseverity = 0.5\n")), None).unwrap_err().contains("dropout_rate"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------ P2

    /// A compat-legal sidecar: simasl's default ASL series on a small grid.
    fn compat_base() -> Value {
        let mut s = base();
        s["M0Type"] = json!("Included");
        s["RepetitionTimePreparation"] = json!([10.0, 5.0, 5.0]);
        s["SliceTiming"] = json!([0.0, 0.0, 0.0]);
        s
    }
    const COMPAT_CTX: &str = "volume_type\nm0scan\ncontrol\nlabel\n";

    #[test]
    fn compat_pins_the_acquisition_with_an_otherwise_empty_overlay() {
        let p = parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\nasldro = true\n")), None).unwrap();
        assert_eq!(p.compat, Some(CompatSpec { desired_snr: None }));
        assert_eq!(p.grid_origin, GridOrigin::VoxelCentre);
        let a = &p.acq;
        assert_eq!((a.oversample, a.n_coils, a.n_spikes), (1, 1, 0));
        assert_eq!((a.partial_fourier, a.ghost_offset, a.signal_scale, a.noise_variance), (1.0, 0.0, 1.0, 0.0));
        assert_eq!((a.eddy_strength, a.eddy_quad, a.eddy_phase), (0.0, 0.0, 0.0));
        assert_eq!(a.window, KspaceWindow::None);
        let acq = p.acquisition(8, 8).unwrap();
        assert!(!acq.do_relaxation && acq.accel == 1 && acq.signal_scale == 1.0);
        // explicit values equal to the pins are fine; SNR 0 is no noise
        let ov = "[compat]\nasldro = true\ndesired_snr = 0.0\n[acquisition]\noversample = 1\nsignal_scale = 1.0\nwindow = \"none\"\n";
        let p = parse(&compat_base(), COMPAT_CTX, Some(&overlay(ov)), None).unwrap();
        assert_eq!(p.compat, Some(CompatSpec { desired_snr: None }));
        let p = parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\nasldro = true\ndesired_snr = 50.0\n")), None).unwrap();
        assert_eq!(p.compat, Some(CompatSpec { desired_snr: Some(50.0) }));
        // without compat: nothing changes, relaxation stays on, the origin is the P1 corner
        let p = parse(&compat_base(), COMPAT_CTX, None, None).unwrap();
        assert!(p.compat.is_none() && p.grid_origin == GridOrigin::Corner && p.acq == OverlayAcq::default());
        assert!(p.acquisition(8, 8).unwrap().do_relaxation);
    }

    #[test]
    fn compat_rejects_each_conflicting_explicit_key_naming_both_values() {
        for (key, bad, pinned) in [
            ("oversample", "2", "1"), ("partial_fourier", "0.75", "1"), ("n_coils", "4", "1"),
            ("ghost_offset", "0.1", "0"), ("n_spikes", "3", "0"), ("eddy_strength", "0.5", "0"),
            ("eddy_quad", "0.5", "0"), ("eddy_phase", "0.5", "0"), ("signal_scale", "100.0", "1"),
            ("noise_variance", "4.0", "0"),
        ] {
            let ov = format!("[compat]\nasldro = true\n[acquisition]\n{key} = {bad}\n");
            let e = parse(&compat_base(), COMPAT_CTX, Some(&overlay(&ov)), None).unwrap_err();
            assert!(e.contains(key) && e.contains(&format!("pins it to {pinned}")), "{key}: {e}");
        }
        let e = parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\nasldro = true\n[acquisition]\nwindow = \"hann\"\n")), None).unwrap_err();
        assert!(e.contains("window") && e.contains("\"none\""), "{e}");
    }

    #[test]
    fn compat_refuses_what_simasl_cannot_express() {
        let ov = overlay("[compat]\nasldro = true\n");
        let check = |s: &Value, ctx: &str, ov: &Overlay, want: &str| {
            let e = parse(s, ctx, Some(ov), None).unwrap_err();
            assert!(e.contains(want) && e.contains("simasl"), "{want}: {e}");
        };
        let mut s = compat_base();
        s["ParallelReductionFactorInPlane"] = json!(2);
        check(&s, COMPAT_CTX, &ov, "ParallelReductionFactorInPlane");
        let mut s = compat_base();
        s["PartialFourier"] = json!(0.75);
        check(&s, COMPAT_CTX, &ov, "PartialFourier");
        s["PartialFourier"] = json!(1);
        assert!(parse(&s, COMPAT_CTX, Some(&ov), None).is_ok());
        let mut s = compat_base();
        s["MultibandAccelerationFactor"] = json!(3);
        check(&s, COMPAT_CTX, &ov, "MultibandAccelerationFactor");
        let mut s = compat_base();
        s["BackgroundSuppression"] = json!(true);
        s["BackgroundSuppressionNumberPulses"] = json!(0);
        s["BackgroundSuppressionPulseTime"] = json!([]);
        check(&s, COMPAT_CTX, &ov, "BackgroundSuppression");
        let mut s = compat_base();
        s["MultibandAccelerationFactor"] = json!(3);
        s["SliceTiming"] = json!([0.0, 0.0, 0.0]);
        let wv = overlay("[compat]\nasldro = true\n[motion.within_volume]\ndropout_rate = 0.2\nseverity = 0.5\n");
        let e = parse(&s, COMPAT_CTX, Some(&wv), None).unwrap_err();
        assert!(e.contains("simasl"), "{e}");
        let mut s = compat_base();
        s["M0Type"] = json!("Separate");
        s["RepetitionTimePreparation"] = json!(5.0);
        let sep = overlay("[compat]\nasldro = true\n[m0]\nrepetition_time = 10.0\n");
        check(&s, "volume_type\ncontrol\nlabel\n", &sep, "Separate");
        let mut s = compat_base();
        s["SliceTiming"] = json!([0.0, 0.05, 0.10]);
        check(&s, COMPAT_CTX, &ov, "SliceTiming");
        // equal but nonzero entries are zero offsets
        s["SliceTiming"] = json!([0.02, 0.02, 0.02]);
        assert!(parse(&s, COMPAT_CTX, Some(&ov), None).unwrap().slice_offsets.iter().all(|t| *t == 0.0));
        // between-volume motion stays allowed (benchmark D)
        let mv = overlay("[compat]\nasldro = true\n[motion]\nmode = \"random\"\nrot_deg = [0, 0, 2]\n");
        assert!(parse(&compat_base(), COMPAT_CTX, Some(&mv), None).is_ok());
    }

    #[test]
    fn compat_keys_parse_and_default_per_the_table() {
        let p = parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\ngrid_origin = \"voxel-centre\"\n")), None).unwrap();
        assert!(p.compat.is_none() && p.grid_origin == GridOrigin::VoxelCentre);
        let p = parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\nasldro = true\ngrid_origin = \"corner\"\n")), None).unwrap();
        assert!(p.compat.is_some() && p.grid_origin == GridOrigin::Corner);
        assert!(parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\ngrid_origin = \"centre\"\n")), None).unwrap_err().contains("grid_origin"));
        assert!(parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\ndesired_snr = 50.0\n")), None).unwrap_err().contains("asldro"));
        assert!(parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\nasldro = true\ndesired_snr = -1.0\n")), None).unwrap_err().contains("desired_snr"));
        assert!(toml::from_str::<Overlay>("[compat]\nasldr = true\n").is_err());
        let p = parse(&compat_base(), COMPAT_CTX, Some(&overlay("[compat]\nasldro = false\n")), None).unwrap();
        assert!(p.compat.is_none() && p.grid_origin == GridOrigin::Corner);
    }

    #[test]
    fn the_cli_flag_is_the_overlay_key() {
        let dir = std::env::temp_dir().join(format!("aslscan-compat-flag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (j, c, o) = (dir.join("asl.json"), dir.join("aslcontext.tsv"), dir.join("ov.toml"));
        std::fs::write(&j, compat_base().to_string()).unwrap();
        std::fs::write(&c, COMPAT_CTX).unwrap();
        // no overlay at all: the flag alone
        assert!(load_with(&[j.as_path()], &c, None, None, true).unwrap().compat.is_some());
        assert!(load_with(&[j.as_path()], &c, None, None, false).unwrap().compat.is_none());
        // an overlay saying true, or saying nothing about compat, is fine; false is a conflict
        std::fs::write(&o, "[compat]\nasldro = true\ndesired_snr = 20.0\n").unwrap();
        assert_eq!(load_with(&[j.as_path()], &c, Some(&o), None, true).unwrap().compat, Some(CompatSpec { desired_snr: Some(20.0) }));
        std::fs::write(&o, "seed = 3\n").unwrap();
        assert!(load_with(&[j.as_path()], &c, Some(&o), None, true).unwrap().compat.is_some());
        std::fs::write(&o, "[compat]\nasldro = false\n").unwrap();
        assert!(load_with(&[j.as_path()], &c, Some(&o), None, true).unwrap_err().contains("--compat-asldro"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------ P4

    const M0: &str = "[m0]\nrepetition_time = 8.0\n";
    const TABLES: &str = "[macrovascular]\narterial_blood_volume = { grey_matter = 0.02, white_matter = 0.01, csf = 0.0 }\n\
                          arterial_transit_time = { grey_matter = 0.5, white_matter = 0.7, csf = 0.0 }\n";

    fn p4(s: &Value, ov: &str, ph: Option<PhantomParams>) -> Result<Protocol, String> {
        parse(s, CTX, Some(&overlay(&format!("{M0}{ov}"))), ph.as_ref())
    }

    fn maps(abv: bool, aatt: bool) -> Option<PhantomParams> {
        Some(PhantomParams { has_abv: abv, has_aatt: aatt, ..Default::default() })
    }

    #[test]
    fn p4_inputs_are_off_and_absent_for_p1_p3_protocols() {
        // (pcasl_single and pcasl_multipld are refused by the P3 readout-in-TR check, as before P4)
        for name in ["asl002", "pasl_cutoff", "crop_pcasl"] {
            let (s, ctx) = fixture(name);
            let ov = overlay(M0);
            let p = parse(&s, &ctx, Some(&ov), None).unwrap();
            assert!(p.exchange_time.is_none() && p.macrovascular.is_none() && p.crushing.is_none() && p.physio.is_none());
            if let Some(sup) = &p.suppression {
                assert_eq!(sup.model, SuppressionModel::GlobalBolus);
            }
            // the row clock: cumulative repetition times
            let mut clock = 0.0;
            for (i, r) in p.rows.iter().enumerate() {
                assert_eq!(p.row_start[i], clock);
                clock += r.tr;
            }
        }
    }

    #[test]
    fn the_arterial_compartment_resolves_each_quantity_from_one_source() {
        let s = base();
        // both tables
        let m = p4(&s, TABLES, None).unwrap().macrovascular.unwrap();
        assert!(matches!(m.abv, QuantitySource::Table(_)) && matches!(m.aatt, QuantitySource::Table(_)));
        // both maps
        let m = p4(&s, "", maps(true, true)).unwrap().macrovascular.unwrap();
        assert_eq!((m.abv, m.aatt), (QuantitySource::Map, QuantitySource::Map));
        // the two mixed combinations
        let m = p4(&s, "[macrovascular]\narterial_transit_time = { grey_matter = 0.5 }\n", maps(true, false)).unwrap().macrovascular.unwrap();
        assert!(m.abv == QuantitySource::Map && matches!(m.aatt, QuantitySource::Table(_)));
        let m = p4(&s, "[macrovascular]\narterial_blood_volume = { grey_matter = 0.02 }\n", maps(false, true)).unwrap().macrovascular.unwrap();
        assert!(matches!(m.abv, QuantitySource::Table(_)) && m.aatt == QuantitySource::Map);
        // missing, duplicate, empty, a lone map
        assert!(p4(&s, "[macrovascular]\narterial_blood_volume = { grey_matter = 0.02 }\n", None).unwrap_err().contains("arterial_transit_time has no source"));
        assert!(p4(&s, TABLES, maps(true, false)).unwrap_err().contains("both"));
        assert!(p4(&s, "[macrovascular]\n", None).unwrap_err().contains("no source"));
        assert!(p4(&s, "", maps(true, false)).unwrap_err().contains("arterial_transit_time has no source"));
        // ranges
        assert!(p4(&s, "[macrovascular]\narterial_blood_volume = { grey_matter = 1.5 }\narterial_transit_time = { grey_matter = 0.5 }\n", None)
            .unwrap_err().contains("out of range"));
        assert!(p4(&s, "[macrovascular]\narterial_blood_volume = { grey_matter = 0.1 }\narterial_transit_time = { grey_matter = -0.5 }\n", None)
            .unwrap_err().contains("out of range"));
        // no P4 input, no maps: off
        assert!(p4(&s, "", maps(false, false)).unwrap().macrovascular.is_none());
    }

    #[test]
    fn the_arterial_t2_inherits_the_resolved_blood_t2() {
        let s = base();
        let m = p4(&s, TABLES, None).unwrap().macrovascular.unwrap();
        assert_eq!(m.t2_arterial, (0.165, Source::T2Blood));
        let m = p4(&s, &format!("{TABLES}[signal]\nt2_blood = 0.2\n"), None).unwrap().macrovascular.unwrap();
        assert_eq!(m.t2_arterial, (0.2, Source::T2Blood));
        let m = p4(&s, &format!("{TABLES}[signal]\nt2_arterial = 0.25\n"), None).unwrap().macrovascular.unwrap();
        assert_eq!(m.t2_arterial, (0.25, Source::Overlay));
        assert!(p4(&s, "[signal]\nt2_arterial = 0.25\n", None).unwrap_err().contains("t2_arterial without"));
        assert!(p4(&s, &format!("{TABLES}[signal]\nt2_arterial = 0.0\n"), None).unwrap_err().contains("t2_arterial"));
    }

    #[test]
    fn crushing_follows_the_combination_matrix() {
        let crushed = |venc: Value| {
            let mut s = base();
            s["VascularCrushing"] = json!(true);
            s["VascularCrushingVENC"] = venc;
            s
        };
        let vel = "[vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 0.0 }\n";
        // with the arterial compartment and velocities: on, VENC per row
        let c = p4(&crushed(json!(4.0)), &format!("{TABLES}{vel}"), None).unwrap().crushing.unwrap();
        assert_eq!(c.venc, vec![4.0; 4]);
        assert!(!c.no_arterial_compartment && c.arterial_velocity.is_some());
        let c = p4(&crushed(json!([0.0, 4.0, 0.0, 4.0])), &format!("{TABLES}{vel}"), None).unwrap().crushing.unwrap();
        assert_eq!(c.venc, vec![0.0, 4.0, 0.0, 4.0]);
        // the refusals
        let mut no_venc = base();
        no_venc["VascularCrushing"] = json!(true);
        assert!(p4(&no_venc, &format!("{TABLES}{vel}"), None).unwrap_err().contains("VascularCrushingVENC"));
        assert!(p4(&crushed(json!(4.0)), TABLES, None).unwrap_err().contains("arterial_velocity"));
        assert!(p4(&crushed(json!(4.0)), &format!("{TABLES}[vascular_crushing]\narterial_velocity = {{ grey_matter = 1.0 }}\nno_arterial_compartment = true\n"), None)
            .unwrap_err().contains("contradicts"));
        assert!(p4(&crushed(json!(4.0)), "", None).unwrap_err().contains("[macrovascular]"));
        assert!(p4(&crushed(json!(4.0)), "[vascular_crushing]\nno_arterial_compartment = true\narterial_velocity = { grey_matter = 1.0 }\n", None)
            .unwrap_err().contains("act on nothing"));
        assert!(p4(&crushed(json!(0.05)), &format!("{TABLES}{vel}"), None).unwrap_err().contains("0.1"));
        assert!(p4(&crushed(json!([4.0, 4.0])), &format!("{TABLES}{vel}"), None).unwrap_err().contains("entries"));
        // velocities above 10 m/s are refused (the review's 1e308 overflowed the crusher ratio);
        // the bound itself is accepted
        for (v, ok) in [("1e308", false), ("1000.5", false), ("1000.0", true)] {
            let r = p4(&crushed(json!(0.1)), &format!("{TABLES}[vascular_crushing]\narterial_velocity = {{ grey_matter = {v} }}\n"), None);
            assert_eq!(r.is_ok(), ok, "{v}: {:?}", r.as_ref().err());
        }
        let mut venc_only = base();
        venc_only["VascularCrushingVENC"] = json!(4.0);
        assert!(p4(&venc_only, "", None).unwrap_err().contains("without VascularCrushing"));
        assert!(p4(&base(), vel, None).unwrap_err().contains("without VascularCrushing"));
        // the escape hatch: accepted, recorded, nothing to act on
        let c = p4(&crushed(json!(4.0)), "[vascular_crushing]\nno_arterial_compartment = true\n", None).unwrap().crushing.unwrap();
        assert!(c.no_arterial_compartment && c.arterial_velocity.is_none());
        // VascularCrushing false stays what P1 accepted
        let mut off = base();
        off["VascularCrushing"] = json!(false);
        assert!(p4(&off, "", None).unwrap().crushing.is_none());
    }

    #[test]
    fn the_suppression_model_and_its_region_keys() {
        let bp = |region: &str| format!("[background_suppression]\nmodel = \"bolus-position\"\n{region}");
        let s = with_suppression(&[2.0, 3.2]);
        let model = |ov: &str| p4(&s, ov, None).map(|p| p.suppression.unwrap().model);
        assert_eq!(model("").unwrap(), SuppressionModel::GlobalBolus);
        assert_eq!(model(&bp("pulse_region = \"global\"\n")).unwrap(), SuppressionModel::BolusPosition(Region::Global));
        assert_eq!(model(&bp("pulse_region = \"slab\"\nslab_entry_time = 0.4\n")).unwrap(), SuppressionModel::BolusPosition(Region::Slab(0.4)));
        assert_eq!(model(&bp("pulse_region = \"slab\"\nslab_entry_time = \"arrival\"\n")).unwrap(), SuppressionModel::BolusPosition(Region::Arrival));
        assert!(model(&bp("")).unwrap_err().contains("needs pulse_region"));
        assert!(model(&bp("pulse_region = \"slab\"\n")).unwrap_err().contains("needs slab_entry_time"));
        assert!(model(&bp("pulse_region = \"global\"\nslab_entry_time = 0.4\n")).unwrap_err().contains("only with"));
        assert!(model(&bp("pulse_region = \"slab\"\nslab_entry_time = -0.4\n")).unwrap_err().contains("slab_entry_time"));
        assert!(model(&bp("pulse_region = \"slab\"\nslab_entry_time = \"later\"\n")).unwrap_err().contains("arrival"));
        assert!(model(&bp("pulse_region = \"brain\"\n")).unwrap_err().contains("pulse_region"));
        assert!(model("[background_suppression]\nmodel = \"other\"\n").unwrap_err().contains("model"));
        assert!(model("[background_suppression]\npulse_region = \"global\"\n").unwrap_err().contains("no effect"));
        // model keys without suppression
        assert!(p4(&base(), &bp("pulse_region = \"global\"\n"), None).unwrap_err().contains("BackgroundSuppression true"));
    }

    #[test]
    fn partial_bolus_inversion_rules() {
        // a pulse at 1.5 s, inside the 1.8 s PCASL bolus
        let early = with_suppression(&[1.5, 3.2]);
        let bp = |region: &str| format!("[background_suppression]\nmodel = \"bolus-position\"\n{region}");
        assert!(p4(&early, "", None).unwrap_err().contains("bolus-position"));
        assert!(p4(&early, &bp("pulse_region = \"slab\"\nslab_entry_time = 0.3\n"), None).is_ok());
        assert!(p4(&early, &bp("pulse_region = \"slab\"\nslab_entry_time = \"arrival\"\n"), None).is_ok());
        assert!(p4(&early, &bp("pulse_region = \"global\"\n"), None).unwrap_err().contains("inflowing blood"));
        // PASL: a global pulse before the cutoff is allowed
        let mut pasl = with_suppression(&[0.5, 1.5]);
        pasl["ArterialSpinLabelingType"] = json!("PASL");
        pasl["BolusCutOffFlag"] = json!(true);
        pasl["BolusCutOffTechnique"] = json!("Q2TIPS");
        pasl["BolusCutOffDelayTime"] = json!(0.7);
        pasl.as_object_mut().unwrap().remove("LabelingDuration");
        assert!(p4(&pasl, "", None).is_err(), "global-bolus still refuses it");
        assert!(p4(&pasl, &bp("pulse_region = \"global\"\n"), None).is_ok());
    }

    #[test]
    fn physio_and_exchange_inputs() {
        let s = base();
        let p = p4(&s, "[physio]\ntissue_cardiac = 0.01\nlabel_drift = 0.02\n", None).unwrap();
        let ph = p.physio.unwrap();
        assert_eq!(ph.tissue, [0.01, 0.0, 0.0]);
        assert_eq!(ph.label, [0.0, 0.0, 0.02]);
        assert_eq!((ph.cardiac_frequency, ph.respiratory_frequency, ph.drift_time), (1.0, 0.25, 30.0));
        assert!(p4(&s, "[physio]\ncardiac_frequency = 1.1\n", None).unwrap_err().contains("all six amplitudes zero"));
        assert!(p4(&s, "[physio]\ntissue_cardiac = 0.01\ncardiac_cv = 0.4\n", None).unwrap_err().contains("cardiac_cv"));
        assert!(p4(&s, "[physio]\ntissue_cardiac = 0.01\ndrift_time = 0.0\n", None).unwrap_err().contains("drift_time"));
        assert!(p4(&s, "[physio]\ntissue_cardiac = 0.01\nrespiratory_frequency = -1.0\n", None).unwrap_err().contains("respiratory_frequency"));
        // frequencies whose period is infinite (the final review's 1e-320: NaN output) or so
        // short that a series holds unboundedly many are refused; the range ends are accepted
        for (f, ok) in [("1e-320", false), ("0.009", false), ("0.01", true), ("10.0", true), ("10.5", false), ("1e300", false)] {
            let r = p4(&s, &format!("[physio]\ntissue_cardiac = 0.01\ncardiac_frequency = {f}\n"), None);
            assert_eq!(r.is_ok(), ok, "{f}: {:?}", r.as_ref().err());
        }
        // exchange times whose reciprocal overflows are refused (the review's 1e-320 read as
        // slow exchange); a microsecond is accepted
        for (x, ok) in [("1e-320", false), ("9e-7", false), ("1e-6", true)] {
            let r = p4(&s, &format!("[kinetic]\nexchange_time = {x}\n"), None);
            assert_eq!(r.is_ok(), ok, "{x}: {:?}", r.as_ref().err());
        }
        // bounded periodic amplitudes
        assert!(p4(&s, "[physio]\ntissue_cardiac = 0.6\ntissue_respiratory = -0.4\n", None).unwrap_err().contains("change sign"));
        assert!(p4(&s, "[physio]\nlabel_cardiac = 1.5\n", None).unwrap_err().contains("label"));
        assert!(p4(&s, "[physio]\ntissue_cardiac = 0.6\ntissue_respiratory = 0.3\ntissue_drift = 2.0\n", None).is_ok());
        assert_eq!(p4(&s, "[kinetic]\nexchange_time = 0.5\n", None).unwrap().exchange_time, Some(0.5));
        assert!(p4(&s, "[kinetic]\nexchange_time = 0.0\n", None).unwrap_err().contains("exchange_time"));
    }

    #[test]
    fn every_p4_part_is_refused_under_compat() {
        let s = compat_base();
        let ov = |extra: &str| overlay(&format!("[compat]\nasldro = true\n{extra}"));
        let check = |s: &Value, extra: &str, ph: Option<PhantomParams>, want: &str| {
            let e = parse(s, COMPAT_CTX, Some(&ov(extra)), ph.as_ref()).unwrap_err();
            assert!(e.contains(want) && e.contains("simasl"), "{want}: {e}");
        };
        check(&s, "[kinetic]\nexchange_time = 0.5\n", None, "part A");
        check(&s, TABLES, None, "part B");
        check(&s, "", maps(true, true), "part B");
        let mut c = compat_base();
        c["VascularCrushing"] = json!(true);
        c["VascularCrushingVENC"] = json!(4.0);
        check(&c, "[vascular_crushing]\nno_arterial_compartment = true\n", None, "part C");
        check(&s, "[physio]\ntissue_cardiac = 0.01\n", None, "part E");
        // part D needs suppression, which compat refuses first (P2); the order is fine either way
        let mut b = compat_base();
        b["BackgroundSuppression"] = json!(true);
        b["BackgroundSuppressionNumberPulses"] = json!(0);
        b["BackgroundSuppressionPulseTime"] = json!([]);
        let e = parse(&b, COMPAT_CTX, Some(&ov("[background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"global\"\n")), None).unwrap_err();
        assert!(e.contains("simasl"), "{e}");
    }

    fn overlay_file(name: &str) -> Overlay {
        let d = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/");
        toml::from_str(&std::fs::read_to_string(format!("{d}{name}/overlay.toml")).unwrap()).unwrap()
    }

    /// The acceptance fixtures resolve to the reviewed numbers (P5 plan, Tasks 4, 7 and 10).
    #[test]
    fn the_grase_acceptance_fixtures_resolve_to_the_reviewed_numbers() {
        // asl005: the real sidecar, its overlay; 64 x 64 x 30
        let (s, c) = fixture("asl005");
        let p = parse(&s, &c, Some(&overlay_file("asl005_p5")), None).unwrap();
        let r = p.readout.as_ref().unwrap();
        assert_eq!((r.kind.0, r.number_shots, r.ky_segments.0, r.refocusing_flip_deg), (ReadoutKind::Grase, (4, Source::Sidecar), 4, (130.0, Source::Sidecar)));
        assert!(p.slice_offsets.is_empty() && p.mb == 1);
        assert_eq!(p.phase_encoding_direction, "j-");
        let res = resolve_readout(&p, [64, 64, 30]).unwrap().unwrap();
        assert_eq!((res.epi, res.etl, res.e_c, res.n_shots), (16, 30, 1, 4));
        assert!((res.t_line_ms - 0.2048).abs() < 1e-12 && res.t_line_source.contains("DwellTime"), "{res:?}");
        assert!((res.t_kyc_ms + 0.1024).abs() < 1e-12 && (res.esp_ms - 13.3824).abs() < 1e-9, "{res:?}");
        // a volume is four shots, each a repetition
        assert!((p.row_start[1] - 4.0 * p.rows[0].tr).abs() < 1e-12);
        // the asl003 derivative: 24 x 20 x 30, two segments, 1 ms lines from EffectiveEchoSpacing
        let (s, c) = fixture("asl003_p5");
        let p = parse(&s, &c, Some(&overlay_file("asl003_p5")), None).unwrap();
        assert_eq!(p.rows.len(), 16);
        let res = resolve_readout(&p, [24, 20, 30]).unwrap().unwrap();
        assert_eq!((res.epi, res.etl, res.n_shots), (10, 30, 2));
        assert!((res.t_line_ms - 1.0).abs() < 1e-12 && res.t_line_source.contains("effective"), "{res:?}");
        assert!((res.t_kyc_ms + 0.5).abs() < 1e-12 && (res.esp_ms - 12.42).abs() < 1e-9, "{res:?}");
        // at the sidecar's 64 lines the block cannot fit, and the message says what would
        let e = resolve_readout(&p, [24, 64, 30]).unwrap_err();
        assert!(e.contains("echo spacing must be at least"), "{e}");
        // an indivisible matrix names the override that would divide
        let e = resolve_readout(&p, [24, 21, 30]).unwrap_err();
        assert!(e.contains("[acquisition] matrix") && e.contains("20"), "{e}");
        // 2D protocols resolve to nothing
        assert!(resolve_readout(&parse(&base(), CTX, Some(&m0_overlay()), None).unwrap(), [64, 64, 3]).unwrap().is_none());
    }

    /// asl001 (a GE 3D spiral) with its overlay resolves to the reviewed numbers (P5 part C, plan
    /// Task 14), and the spiral rules refuse what they must.
    #[test]
    fn the_spiral_acceptance_fixture_resolves_and_the_spiral_rules_hold() {
        let (s, c) = fixture("asl001");
        let p = parse(&s, &c, Some(&overlay_file("asl001_p5")), None).unwrap();
        let r = p.readout.as_ref().unwrap();
        assert_eq!((r.kind, r.number_shots, r.interleaves), ((ReadoutKind::Spiral, Source::Sidecar), (8, Source::Default),
                                                             Some((8, Source::Overlay))));
        assert_eq!(r.dwell_time_s, Some((4e-6, Source::Overlay)));
        assert_eq!(r.refocusing_flip_deg, (111.0, Source::Sidecar));
        // a volume is eight shots, each a repetition: 39.088 s
        assert!((p.row_start[1] - 8.0 * 4.886).abs() < 1e-9, "{}", p.row_start[1]);
        if cfg!(feature = "kspace") {
            let res = resolve_readout(&p, [64, 64, 20]).unwrap().unwrap();
            let sp = res.spiral.as_ref().unwrap();
            assert_eq!((res.etl, res.e_c, res.n_shots, sp.interleaves, sp.samples_per_interleaf), (20, 1, 8, 8, 1000));
            assert!((res.esp_ms - 10.528).abs() < 1e-9 && (sp.dwell_ms - 0.004).abs() < 1e-15, "{res:?}");
            assert!(sp.tau_c_ms > 0.0 && sp.tau_c_ms < 0.1, "{}", sp.tau_c_ms);
            assert_eq!(res.readout, Readout3d::Spiral { interleaves: 8, readout_ms: 4.0, dwell_ms: sp.dwell_ms, radial_oversampling: 1.2 });
            assert_eq!(sp.radial_oversampling, (1.2, Source::Default));
            // a rectangular matrix
            assert!(resolve_readout(&p, [64, 48, 20]).unwrap_err().contains("square"));
            // a dwell time past the sampling bound (4.97 us here)
            let ov = |extra: &str| {
                let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/asl001_p5/overlay.toml")).unwrap();
                overlay(&base.replace("dwell_time = 4e-6\n", extra))
            };
            let p6 = parse(&s, &c, Some(&ov("dwell_time = 6e-6\n")), None).unwrap();
            assert!(resolve_readout(&p6, [64, 64, 20]).unwrap_err().contains("dwell time must be below"));
            // an 8 ms spiral would run through the next refocusing pulse at this echo spacing
            let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/asl001_p5/overlay.toml")).unwrap();
            let ov8 = overlay(&base.replace("interleaves = 8\nspiral_readout_time = 4.0", "interleaves = 4\nspiral_readout_time = 8.0"));
            let p8 = parse(&s, &c, Some(&ov8), None).unwrap();
            assert!(resolve_readout(&p8, [64, 64, 20]).unwrap_err().contains("next refocusing pulse"));
        } else {
            assert!(resolve_readout(&p, [64, 64, 20]).unwrap_err().contains("kspace"));
        }
        // the parse rules: required values, NumberShots, refused keys and effects
        let with = |extra: &str| {
            let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/asl001_p5/overlay.toml")).unwrap();
            overlay(&format!("{base}{extra}"))
        };
        let strip = |key: &str| {
            let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/asl001_p5/overlay.toml")).unwrap();
            overlay(&base.lines().filter(|l| !l.starts_with(key)).collect::<Vec<_>>().join("\n"))
        };
        assert!(parse(&s, &c, Some(&strip("spiral_readout_time")), None).unwrap_err().contains("spiral_readout_time"));
        assert!(parse(&s, &c, Some(&strip("dwell_time")), None).unwrap_err().contains("DwellTime"));
        let mut s2 = s.clone();
        s2["DwellTime"] = json!(4e-6);
        assert_eq!(parse(&s2, &c, Some(&strip("dwell_time")), None).unwrap().readout.unwrap().dwell_time_s, Some((4e-6, Source::Sidecar)));
        s2["NumberShots"] = json!(4);
        assert!(parse(&s2, &c, Some(&with("")), None).unwrap_err().contains("NumberShots 4"));
        s2["NumberShots"] = json!(8);
        assert_eq!(parse(&s2, &c, Some(&with("")), None).unwrap().readout.unwrap().number_shots, (8, Source::Sidecar));
        // a zero arterial blood T1 cannot drive the echo amplitudes (before any feature check)
        let p0 = parse(&s, &c, Some(&with("[kinetic]\nt1_arterial_blood = 0.0\n")), None).unwrap();
        assert!(resolve_readout(&p0, [64, 64, 20]).unwrap_err().contains("arterial blood T1"));
        for (key, val) in [("PhaseEncodingDirection", json!("j-")), ("PhaseEncodingDirection", json!(["j-"])),
                           ("PhaseEncodingDirection", json!(17)), ("TotalReadoutTime", json!(0.03)),
                           ("EffectiveEchoSpacing", json!(0.0005))] {
            let mut s3 = s.clone();
            s3[key] = val;
            assert!(parse(&s3, &c, Some(&with("")), None).is_err(), "{key}");
        }
        let mut s3 = s.clone();
        s3["ParallelReductionFactorInPlane"] = json!(2);
        assert!(parse(&s3, &c, Some(&with("")), None).unwrap_err().contains("GRAPPA"));
        for knob in ["partial_fourier = 0.75", "ghost_offset = 0.1", "n_spikes = 2"] {
            let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/asl001_p5/overlay.toml")).unwrap();
            let ov = overlay(&base.replace("matrix = [64, 64]", &format!("matrix = [64, 64]\n{knob}")));
            assert!(parse(&s, &c, Some(&ov), None).unwrap_err().contains("no spiral meaning"), "{knob}");
        }
        assert!(parse(&s, &c, Some(&with("ky_segments = 2\n")), None).unwrap_err().contains("GRASE key"));
        // the radial oversampling: at least 1, the overlay's when given
        assert!(parse(&s, &c, Some(&with("radial_oversampling = 0.9\n")), None).unwrap_err().contains("at least 1"));
        assert_eq!(parse(&s, &c, Some(&with("radial_oversampling = 1.5\n")), None).unwrap().readout.unwrap().radial_oversampling,
                   Some((1.5, Source::Overlay)));
    }

    /// A small GRASE sidecar for the refusal and resolution rules.
    fn grase() -> Value {
        json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [4.0, 4.0, 4.0],
            "MRAcquisitionType": "3D", "PulseSequenceType": "3Dgrase", "PhaseEncodingDirection": "j-",
            "EffectiveEchoSpacing": 0.0003, "NumberShots": 2, "FlipAngle": 150
        })
    }

    // ---- P7 part C: the 3D gradient-echo train ("epi3d")

    fn epi3d() -> Value {
        let mut s = grase();
        s["PulseSequenceType"] = json!("3D EPI");
        s["FlipAngle"] = json!(12);
        s
    }

    const EPI3D: &str = "[signal]\nacq_contrast = \"ge\"\n[readout]\nexcitation_spacing = 50.0\nslab_entry_time = 0.5\n";

    /// The `"epi3d"` inputs: activation, defaults, the slab entry's two sources, every refusal, the
    /// memory key, multi-TE, and the resolution with the grid.
    #[test]
    fn epi3d_protocols() {
        let ok = |s: &Value, ov: &str| parse(s, CTX, Some(&overlay(ov)), None);
        let err = |s: &Value, ov: &str| ok(s, ov).unwrap_err();
        let p = ok(&epi3d(), EPI3D).unwrap();
        let r = p.readout.as_ref().unwrap();
        let g = p.ge3d().unwrap();
        assert_eq!((r.kind, r.ky_segments.0, r.kz_segments.0), ((ReadoutKind::Epi3d, Source::Sidecar), 2, 1));
        assert_eq!((g.excitation_spacing_ms, g.excitation_time_ms, g.max_t1_groups, g.node_tolerance, g.max_memory_gib),
                   (50.0, (2.0, Source::Default), (16, Source::Default), (1e-4, Source::Default), (4.0, Source::Default)));
        assert_eq!(g.slab_entry, (SlabEntryTime::Seconds(0.5), "overlay readout.slab_entry_time"));
        // FlipAngle is the excitation under "epi3d"
        assert_eq!(p.ge.as_ref().unwrap().flip_deg, 12.0);
        assert!(p.p6_active());
        // the overlay's type over a GRASE sidecar
        let po = ok(&grase(), &format!("{EPI3D}type = \"epi3d\"\n")).unwrap_err();
        assert!(po.contains("(0, 90]") && po.contains("refocusing"), "{po}");
        let mut g12 = grase();
        g12["FlipAngle"] = json!(12);
        let pt = ok(&g12, &format!("{EPI3D}type = \"epi3d\"\n")).unwrap();
        assert_eq!(pt.readout.unwrap().kind, (ReadoutKind::Epi3d, Source::Overlay));
        // the slab entry from bolus-position, its "arrival"; both sources refused; neither refused
        let mut bs = epi3d();
        bs["BackgroundSuppression"] = json!(true);
        bs["BackgroundSuppressionNumberPulses"] = json!(1);
        bs["BackgroundSuppressionPulseTime"] = json!([2.0]);
        let bp = "[signal]\nacq_contrast = \"ge\"\n[readout]\nexcitation_spacing = 50.0\n\
                  [background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = \"arrival\"\n";
        assert_eq!(ok(&bs, bp).unwrap().ge3d().unwrap().slab_entry,
                   (SlabEntryTime::Arrival, "overlay background_suppression.slab_entry_time"));
        let both = bp.replace("excitation_spacing = 50.0\n", "excitation_spacing = 50.0\nslab_entry_time = 0.3\n");
        assert!(err(&bs, &both).contains("one source"), "{}", err(&bs, &both));
        let none = "[signal]\nacq_contrast = \"ge\"\n[readout]\nexcitation_spacing = 50.0\n";
        assert!(err(&epi3d(), none).contains("slab entry"));
        // refusals
        assert!(err(&epi3d(), "[readout]\nexcitation_spacing = 50.0\nslab_entry_time = 0.5\n").contains("acq_contrast"));
        assert!(err(&grase(), "[signal]\nacq_contrast = \"ge\"\n").contains("epi3d"));
        assert!(err(&epi3d(), "[signal]\nacq_contrast = \"ge\"\n[readout]\nslab_entry_time = 0.5\n").contains("excitation_spacing"));
        assert!(err(&epi3d(), &format!("{EPI3D}echo_spacing = 10.0\n")).contains("spin-echo train key"));
        assert!(err(&epi3d(), &format!("{EPI3D}interleaves = 4\n")).contains("spiral key"));
        assert!(err(&grase(), "[readout]\nexcitation_spacing = 50.0\n").contains("3D EPI key"));
        let mut nofa = epi3d();
        nofa.as_object_mut().unwrap().remove("FlipAngle");
        assert!(err(&nofa, EPI3D).contains("FlipAngle"));
        assert!(err(&epi3d(), &format!("{EPI3D}node_tolerance = 2.0\n")).contains("(0, 1)"));
        assert!(err(&epi3d(), &format!("{EPI3D}max_t1_groups = 0\n")).contains("at least 1"));
        assert!(err(&epi3d(), &EPI3D.replace("0.5", "\"sometimes\"")).contains("arrival"));
        assert!(err(&epi3d(), &format!("{EPI3D}[acquisition]\npartial_fourier = 0.75\n")).contains("partial Fourier"));
        // the memory key: [images] with "epi3d"; refused with neither epi3d nor compat multi-TE
        assert_eq!(ok(&epi3d(), &format!("{EPI3D}[images]\nmax_memory_gib = 2.0\n")).unwrap().ge3d().unwrap().max_memory_gib,
                   (2.0, Source::Overlay));
        assert!(err(&grase(), "[images]\nmax_memory_gib = 2.0\n").contains("[images]"));
        // multi-TE: several echoes per excitation; GRASE still refuses it
        let two: Vec<Value> = [0.012, 0.030].iter().map(|te| { let mut x = epi3d(); x["EchoTime"] = json!(te); x }).collect();
        let pm = parse_echoes(&two, CTX, Some(&overlay(EPI3D)), None).unwrap();
        assert_eq!(pm.echo_times_s, vec![0.012, 0.030]);
        let twog: Vec<Value> = [0.012, 0.030].iter().map(|te| { let mut x = grase(); x["EchoTime"] = json!(te); x }).collect();
        assert!(parse_echoes(&twog, CTX, None, None).unwrap_err().contains("multi-echo spin-echo"));

        // resolution with the grid: 32 x 32 x 8, two ky segments; EffectiveEchoSpacing 0.3 ms gives
        // lines 0.6 ms apart; the echo is centred so the centre line is read at EchoTime
        let res = resolve_ge3d(&p, [32, 32, 8]).unwrap().unwrap();
        assert!((res.t_line_ms - 0.6).abs() < 1e-12 && res.n_exc == 8 && res.n_shots == 2 && res.epi == 16);
        assert!((res.train.echo_times_ms[0] + res.t_kyc_ms - 12.0).abs() < 1e-12);
        assert!(resolve_readout(&p, [32, 32, 8]).unwrap_err().contains("resolve_ge3d"));
        assert!(resolve_ge3d(&p, [32, 32, 1]).unwrap_err().contains("2D"));
        // a spacing the echo block does not fit in, named with its row
        let tight = ok(&epi3d(), &EPI3D.replace("50.0", "12.0")).unwrap();
        let e = resolve_ge3d(&tight, [32, 32, 8]).unwrap_err();
        assert!(e.contains("row 0") && e.contains("next excitation pulse"), "{e}");
        // the line spacing from DwellTime and readout_samples when nothing else gives it, as GRASE's
        // (final review 6): 64 x 5 us = 0.32 ms, a lower bound
        let mut nosp = epi3d();
        nosp.as_object_mut().unwrap().remove("EffectiveEchoSpacing");
        nosp.as_object_mut().unwrap().remove("TotalReadoutTime");
        assert!(resolve_ge3d(&ok(&nosp, EPI3D).unwrap(), [32, 32, 8]).unwrap_err().contains("line spacing"));
        nosp["DwellTime"] = json!(5e-6);
        let dw = resolve_ge3d(&ok(&nosp, &format!("{EPI3D}readout_samples = 64\n")).unwrap(), [32, 32, 8]).unwrap().unwrap();
        assert!((dw.t_line_ms - 0.32).abs() < 1e-12 && dw.t_line_source.contains("lower bound"), "{}", dw.t_line_ms);
        // a GRASE protocol has no 3D EPI resolution
        assert!(resolve_ge3d(&ok(&grase(), "").unwrap(), [32, 32, 8]).unwrap().is_none());
    }

    #[test]
    fn three_d_inputs_activation_and_refusals() {
        let ok = |s: &Value, ov: &str| parse(s, CTX, Some(&overlay(ov)), None);
        let err = |s: &Value, ov: &str| parse(s, CTX, Some(&overlay(ov)), None).unwrap_err();
        let p = ok(&grase(), "").unwrap();
        let r = p.readout.as_ref().unwrap();
        assert_eq!((r.kind, r.ky_segments, r.kz_segments, r.kz_order), ((ReadoutKind::Grase, Source::Sidecar), (2, Source::Default),
                   (1, Source::Default), (KzOrder::Centric, Source::Default)));
        assert_eq!(r.refocusing_flip_deg, (150.0, Source::Sidecar));
        assert_eq!(r.refocusing_time_ms, (2.0, Source::Default));
        // the sidecar's FlipAngle is the refocusing angle: the spin-echo 90-degree rule does not apply
        // to it, but an overlay excitation angle other than 90 is still refused
        assert!(err(&grase(), "[signal]\nexcitation_flip_angle = 60\n").contains("90-degree"));
        // the readout type: the overlay over PulseSequenceType; unknown types listed
        let mut s = grase();
        s["PulseSequenceType"] = json!("EPI");
        assert!(err(&s, "").contains("\"grase\""));
        assert_eq!(ok(&s, "[readout]\ntype = \"grase\"\n").unwrap().readout.unwrap().kind, (ReadoutKind::Grase, Source::Overlay));
        // refused with 3D
        for (key, v, want) in [("SliceTiming", json!([0.0, 0.05]), "SliceTiming"), ("SliceEncodingDirection", json!("k"), "SliceEncodingDirection"),
                               ("MultibandAccelerationFactor", json!(2), "Multiband"),
                               ("ParallelReductionFactorOutOfPlane", json!(2), "out-of-plane"),
                               ("PartialFourierDirection", json!("k"), "kz"), ("NumberShots", json!([2, 2]), "NumberShots")] {
            let mut s = grase();
            s[key] = v;
            assert!(err(&s, "").contains(want), "{key}: {}", err(&s, ""));
        }
        assert!(err(&grase(), "[signal]\nacq_contrast = \"ge\"\n").contains("3D gradient-echo"));
        assert!(err(&grase(), "[signal]\nacq_contrast = \"ir\"\n").contains("inversion recovery"));
        assert!(err(&grase(), "[compat]\nasldro = true\n").contains("simasl"));
        assert!(err(&grase(), "[acquisition]\neddy_strength = 0.1\n").contains("eddy"));
        // segmentation must multiply to NumberShots; kz_order names its values
        assert!(err(&grase(), "[readout]\nky_segments = 4\n").contains("NumberShots"));
        assert!(ok(&grase(), "[readout]\nky_segments = 1\nkz_segments = 2\n").is_ok());
        assert!(err(&grase(), "[readout]\nkz_order = \"spiral\"\n").contains("centric"));
        // refocusing range; keys of the other readout; [readout] with 2D
        assert!(err(&grase(), "[readout]\nrefocusing_flip_angle = 190\n").contains("(0, 180]"));
        assert!(err(&grase(), "[readout]\ninterleaves = 8\n").contains("spiral key"));
        assert!(err(&grase(), "[readout]\nradial_oversampling = 1.2\n").contains("spiral key"));
        assert!(parse(&base(), CTX, Some(&overlay("[readout]\nky_segments = 2\n[m0]\nrepetition_time = 8.0\n")), None)
            .unwrap_err().contains("2D"));
        // GRASE needs a phase-encode direction; the overlay supplies one; spirals refuse it
        let mut s = grase();
        s.as_object_mut().unwrap().remove("PhaseEncodingDirection");
        assert!(err(&s, "").contains("phase_encoding_direction"));
        assert!(ok(&s, "[readout]\nphase_encoding_direction = \"j\"\n").unwrap().reverse_phase);
        let mut sp = grase();
        sp["PulseSequenceType"] = json!("spiral");
        sp.as_object_mut().unwrap().remove("EffectiveEchoSpacing");
        assert!(err(&sp, "").contains("spiral"));
        sp.as_object_mut().unwrap().remove("PhaseEncodingDirection");
        assert!(err(&sp, "[readout]\nky_segments = 2\n").contains("GRASE key"));
        // within-volume motion in 3D needs NumberShots > 1
        let mut one = grase();
        one["NumberShots"] = json!(1);
        let wv = "[motion]\nwithin_volume = { dropout_rate = 0.1, severity = 0.5, jump_mm = [0.5, 0.0, 0.0], jump_deg = [0.0, 0.0, 0.0] }\n";
        assert!(err(&one, wv).contains("NumberShots > 1"));
        assert!(ok(&grase(), wv).unwrap().motion.unwrap().within.is_some());
    }

    #[test]
    fn three_d_resolution_rules() {
        let res = |s: &Value, ov: &str, dims: [usize; 3]| resolve_readout(&parse(s, CTX, Some(&overlay(ov)), None).unwrap(), dims);
        // the effective spacing gives the actual line: 0.3 ms x 2 segments
        let r = res(&grase(), "", [32, 32, 20]).unwrap().unwrap();
        assert!((r.t_line_ms - 0.6).abs() < 1e-12, "{r:?}");
        // TotalReadoutTime / (ny - 1) when that is all there is; both must agree when both are given
        let mut s = grase();
        s.as_object_mut().unwrap().remove("EffectiveEchoSpacing");
        s["TotalReadoutTime"] = json!(0.0093);
        assert!((res(&s, "", [32, 32, 20]).unwrap().unwrap().t_line_ms - 0.6).abs() < 1e-12);
        s["EffectiveEchoSpacing"] = json!(0.0004);
        assert!(res(&s, "", [32, 32, 20]).unwrap_err().contains("disagree"));
        // the overlay's line spacing must agree with an effective spacing
        assert!(res(&grase(), "[readout]\nline_spacing = 0.9\n", [32, 32, 20]).unwrap_err().contains("disagrees"));
        // and one accepted within 1% is the one published: 0.605 ms over 2 segments, not 0.0003 s
        let r = res(&grase(), "[readout]\nline_spacing = 0.605\n", [32, 32, 20]).unwrap().unwrap();
        assert!((r.t_line_ms - 0.605).abs() < 1e-15 && (r.effective_spacing_s.unwrap() - 0.0003025).abs() < 1e-15, "{r:?}");
        // no source at all
        let mut s = grase();
        s.as_object_mut().unwrap().remove("EffectiveEchoSpacing");
        assert!(res(&s, "", [32, 32, 20]).unwrap_err().contains("line spacing"));
        // the dwell time, with readout_samples
        s["DwellTime"] = json!(5e-6);
        let r = res(&s, "[readout]\nreadout_samples = 64\n", [32, 32, 20]).unwrap().unwrap();
        assert!((r.t_line_ms - 0.32).abs() < 1e-12 && r.t_line_source.contains("lower bound"));
        // an explicit echo spacing must put the k-space centre at EchoTime
        let r = res(&grase(), "", [32, 32, 20]).unwrap().unwrap();
        let exact = format!("[readout]\necho_spacing = {}\n", r.esp_ms);
        assert!(res(&grase(), &exact, [32, 32, 20]).is_ok());
        assert!(res(&grase(), "[readout]\necho_spacing = 13.0\n", [32, 32, 20]).unwrap_err().contains("EchoTime"));
        // the train must end within the repetition (a long train)
        let mut s = grase();
        s["RepetitionTimePreparation"] = json!(3.7);
        assert!(res(&s, "", [32, 32, 40]).unwrap_err().contains("after"));
        // the separate M0's train is checked at its own repetition time
        let mut s = grase();
        s["M0Type"] = json!("Separate");
        assert!(res(&s, "[m0]\nrepetition_time = 0.2\n", [32, 32, 20]).unwrap_err().contains("after"));
    }

    // ---- P6 part C: multi-TE protocols

    /// `base()` once per echo time.
    fn echoes(tes: &[f64]) -> Vec<Value> {
        tes.iter().map(|te| {
            let mut s = base();
            s["EchoTime"] = json!(te);
            s
        }).collect()
    }

    fn parse_e(s: &[Value], ov: &str) -> Result<Protocol, String> {
        parse_echoes(s, CTX, Some(&overlay(&format!("[m0]\nrepetition_time = 8.0\n{ov}"))), None)
    }

    #[test]
    fn one_sidecar_parses_as_before() {
        let p = parse(&base(), CTX, Some(&m0_overlay()), None).unwrap();
        assert_eq!((p.echo_times_s.clone(), p.multi_te.is_none(), p.p6_active()), (vec![0.012], true, false));
        let q = parse_echoes(&[base()], CTX, Some(&m0_overlay()), None).unwrap();
        assert_eq!((q.echo_times_s, q.rows, q.row_start), (p.echo_times_s, p.rows, p.row_start));
        // an EchoTime array keeps its rule, and the message names the echo-N layout
        let mut s = base();
        s["EchoTime"] = json!([0.012, 0.030]);
        let e = parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err();
        assert!(e.contains("echo-N") && e.contains("--asl-json"), "{e}");
        // [multi_te] needs more than one echo
        assert!(parse_e(&[base()], "[multi_te]\nrefocusing_time = 2.0\n").unwrap_err().contains("more than one echo"));
    }

    #[test]
    fn echo_sidecars_give_the_echo_times() {
        let p = parse_e(&echoes(&[0.012, 0.030, 0.050]), "").unwrap();
        assert_eq!(p.echo_times_s, vec![0.012, 0.030, 0.050]);
        assert_eq!(p.echo_time_s, 0.012);
        assert!(p.p6_active());
        let m = p.multi_te.as_ref().unwrap();
        assert_eq!((m.sidecars.len(), m.refocusing_time_ms, m.max_image_memory_gib), (3, Some((2.0, Source::Default)), None));
        assert_eq!(m.sidecars[1]["EchoTime"], json!(0.030));
        // order: strictly increasing
        assert!(parse_e(&echoes(&[0.030, 0.012]), "").unwrap_err().contains("increase strictly"));
        assert!(parse_e(&echoes(&[0.012, 0.012]), "").unwrap_err().contains("increase strictly"));
        // each echo's EchoTime is a number
        let mut s = echoes(&[0.012, 0.030]);
        s[1]["EchoTime"] = json!([0.030, 0.030]);
        let e = parse_e(&s, "").unwrap_err();
        assert!(e.contains("echo 2") && e.contains("echo-N"), "{e}");
        s[1].as_object_mut().unwrap().remove("EchoTime");
        assert!(parse_e(&s, "").unwrap_err().contains("echo 2"));
        // 3D is refused
        let mut g = echoes(&[0.012, 0.030]);
        for s in &mut g {
            s["MRAcquisitionType"] = json!("3D");
        }
        assert!(parse_e(&g, "").unwrap_err().contains("3D"));
    }

    #[test]
    fn echo_sidecars_must_agree_on_every_other_key() {
        let check = |key: &str, a: Value, b: Value| {
            let mut s = echoes(&[0.012, 0.030]);
            s[0][key] = a.clone();
            s[1][key] = b.clone();
            let e = parse_e(&s, "").unwrap_err();
            assert!(e.contains(key) && e.contains(&a.to_string()) && e.contains(&b.to_string()) && e.contains("echo 2"), "{e}");
        };
        check("RepetitionTimePreparation", json!(4.0), json!(4.5)); // number
        check("SliceTiming", json!([0.0, 0.05, 0.10]), json!([0.0, 0.05, 0.11])); // array
        check("PhaseEncodingDirection", json!("j-"), json!("j")); // string
        check("Custom", json!({"a": 1}), json!({"a": 2})); // object
        // a key in one sidecar only
        let mut s = echoes(&[0.012, 0.030]);
        s[1]["Manufacturer"] = json!("X");
        let e = parse_e(&s, "").unwrap_err();
        assert!(e.contains("Manufacturer") && e.contains("absent"), "{e}");
        // numbers compare as numbers, through arrays and objects
        let mut s = echoes(&[0.012, 0.030]);
        s[0]["RepetitionTimePreparation"] = json!(4);
        s[1]["RepetitionTimePreparation"] = json!(4.0);
        s[0]["Custom"] = json!({"a": [60]});
        s[1]["Custom"] = json!({"a": [60.0]});
        parse_e(&s, "").unwrap();
    }

    #[test]
    fn multi_te_overlay_keys() {
        let two = echoes(&[0.012, 0.030]);
        let p = parse_e(&two, "[multi_te]\nrefocusing_time = 3.0\n").unwrap();
        assert_eq!(p.multi_te.unwrap().refocusing_time_ms, Some((3.0, Source::Overlay)));
        assert!(parse_e(&two, "[multi_te]\nrefocusing_time = 0.0\n").is_err());
        // gradient echo has no refocusing pulse
        let ge = "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n";
        assert_eq!(parse_e(&two, ge).unwrap().multi_te.unwrap().refocusing_time_ms, None);
        assert!(parse_e(&two, &format!("{ge}[multi_te]\nrefocusing_time = 2.0\n")).unwrap_err().contains("gradient"));
        // the image-memory limit is compat's
        assert!(parse_e(&two, "[multi_te]\nmax_image_memory_gib = 2.0\n").unwrap_err().contains("compat"));
        let mut c = echoes(&[0.012, 0.030]);
        for s in &mut c {
            s["M0Type"] = json!("Absent");
            s["SliceTiming"] = json!([0.0, 0.0, 0.0]);
        }
        let cp = parse_echoes(&c, CTX, Some(&overlay("[compat]\nasldro = true\n")), None).unwrap();
        assert_eq!(cp.multi_te.unwrap().max_image_memory_gib, Some((4.0, Source::Default)));
        let cp = parse_echoes(&c, CTX, Some(&overlay("[compat]\nasldro = true\n[multi_te]\nmax_image_memory_gib = 1.5\n")), None).unwrap();
        assert_eq!(cp.multi_te.unwrap().max_image_memory_gib, Some((1.5, Source::Overlay)));
        // P7: [images] max_memory_gib is the same bound under its newer name; both are refused
        let cp = parse_echoes(&c, CTX, Some(&overlay("[compat]\nasldro = true\n[images]\nmax_memory_gib = 2.5\n")), None).unwrap();
        assert_eq!(cp.multi_te.unwrap().max_image_memory_gib, Some((2.5, Source::Overlay)));
        let both = "[compat]\nasldro = true\n[multi_te]\nmax_image_memory_gib = 1.5\n[images]\nmax_memory_gib = 2.5\n";
        assert!(parse_echoes(&c, CTX, Some(&overlay(both)), None).unwrap_err().contains("give one"));
        // compat still refuses a separate M0
        assert!(parse_echoes(&two, CTX, Some(&overlay("[compat]\nasldro = true\n[m0]\nrepetition_time = 8.0\n")), None).is_err());
    }

    /// The half block against the readout `mrsim_acq` builds, whose `half_read_time` is private:
    /// `time_from_max_echo(0) = dt / 2 - h`.
    #[test]
    fn the_half_block_is_the_epi_readout_s() {
        use mrsim_acq::readout::{Readout, SingleShotEpi};
        let p = parse(&base(), CTX, Some(&m0_overlay()), None).unwrap();
        for (nx, ny) in [(64, 64), (24, 20), (37, 53)] {
            let a = p.acquisition(nx, ny).unwrap();
            let epi = SingleShotEpi { kx_max: nx, ky_max: ny, t_line: a.t_line, t_echo: a.t_echo, reverse_phase: false };
            let h_ms = epi.dt() * 0.5 - epi.time_from_max_echo(0);
            let h = half_epi_block_s(&p, nx, ny) * 1000.0;
            assert!((h - h_ms).abs() <= 4.0 * f64::EPSILON * h, "{nx}x{ny}: {h} vs {h_ms}");
        }
    }

    #[test]
    fn excitation_timing_rules() {
        let dims = [32, 32, 3];
        let run = |s: Vec<Value>, ov: &str| check_excitation_timing(&parse_e(&s, ov).unwrap(), dims);
        let ge = "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n";
        // h = TotalReadoutTime / 2 = 8 ms. Gradient echo: blocks [4, 20], [22, 38] ms
        run(echoes(&[0.012, 0.030]), ge).unwrap();
        assert!(run(echoes(&[0.012, 0.025]), ge).unwrap_err().contains("overlap"));
        // spin echo, 2 ms reserved: TE_1 >= 2h + r = 18 ms, TE_2 >= TE_1 + 2h + r
        run(echoes(&[0.020, 0.040]), "").unwrap();
        assert!(run(echoes(&[0.012, 0.040]), "").unwrap_err().contains("first refocusing pulse"));
        assert!(run(echoes(&[0.020, 0.037]), "").unwrap_err().contains("between echoes 1 and 2"));
        assert!(run(echoes(&[0.020, 0.040]), "[multi_te]\nrefocusing_time = 5.0\n").is_err());
        // between excitation groups: offsets 0, 50, 100 ms; the last block ends at TE_2 + 8 ms
        let with = |tes: &[f64], timing: Value, mb: Option<usize>| {
            let mut s = echoes(tes);
            for x in &mut s {
                x["SliceTiming"] = timing.clone();
                if let Some(m) = mb {
                    x["MultibandAccelerationFactor"] = json!(m);
                }
            }
            s
        };
        for timing in [json!([0.0, 0.05, 0.10]), json!([0.10, 0.05, 0.0]), json!([0.0, 0.10, 0.05])] {
            run(with(&[0.020, 0.040], timing.clone(), None), "").unwrap();
            let e = run(with(&[0.020, 0.045], timing.clone(), None), "").unwrap_err();
            assert!(e.contains("next slices' excitation"), "{timing}: {e}");
        }
        // multiband: slices sharing an offset are one group
        let mb = json!([0.0, 0.05, 0.0, 0.05]);
        run(with(&[0.020, 0.040], mb.clone(), Some(2)), "").unwrap();
        assert!(run(with(&[0.020, 0.045], mb, Some(2)), "").unwrap_err().contains("next slices' excitation"));
        // an excitation that fits in TR whose last echo does not: t = 3.6 s, last slice at 3.7 s,
        // its last echo ending at 3.748 s
        let mut s = echoes(&[0.020, 0.040]);
        for x in &mut s {
            x["RepetitionTimePreparation"] = json!(3.72);
        }
        assert!(run(s, "").unwrap_err().contains("RepetitionTimePreparation"));
        // the separate M0 at its own repetition time
        let p = parse_echoes(&echoes(&[0.020, 0.040]), CTX, Some(&overlay("[m0]\nrepetition_time = 0.12\n")), None).unwrap();
        assert!(check_excitation_timing(&p, dims).unwrap_err().contains("separate M0"));
        // an m0scan row excites at its offsets alone
        let mut s = echoes(&[0.020, 0.040]);
        for x in &mut s {
            x["M0Type"] = json!("Included");
            x["PostLabelingDelay"] = json!([0.0, 1.8, 1.8, 1.8, 1.8]);
            x["RepetitionTimePreparation"] = json!([0.16, 4.0, 4.0, 4.0, 4.0]);
        }
        let ctx = "volume_type\nm0scan\ncontrol\nlabel\ncontrol\nlabel\n";
        let p = parse_echoes(&s, ctx, None, None).unwrap();
        check_excitation_timing(&p, dims).unwrap();
        for x in &mut s {
            x["RepetitionTimePreparation"] = json!([0.14, 4.0, 4.0, 4.0, 4.0]);
        }
        let p = parse_echoes(&s, ctx, None, None).unwrap();
        assert!(check_excitation_timing(&p, dims).unwrap_err().contains("row 0"));
        // compat excites every slice together (equal SliceTiming): one group, whose checks run
        let compat = |tes: &[f64]| {
            let mut s = echoes(tes);
            for x in &mut s {
                x["M0Type"] = json!("Absent");
                x["SliceTiming"] = json!([0.0, 0.0, 0.0]);
            }
            check_excitation_timing(&parse_echoes(&s, CTX, Some(&overlay("[compat]\nasldro = true\n")), None).unwrap(), dims)
        };
        compat(&[0.020, 0.045]).unwrap();
        assert!(compat(&[0.020, 0.037]).unwrap_err().contains("between echoes 1 and 2"));
    }

    // ---- P6 part A: Hadamard protocols

    /// A Hadamard protocol of `order` and `cycles` cycles, an m0scan row first: its sidecar,
    /// aslcontext, and the sub-bolus durations given, PLD = `pld`.
    pub(crate) fn hadamard_input(order: usize, cycles: usize, tau: &[f64], pld: f64) -> (Value, String) {
        let n = order - 1;
        assert_eq!(tau.len(), n);
        let mut ld = vec![0.0];
        let mut plds = vec![0.0];
        for _ in 0..cycles {
            for j in 0..n {
                ld.push(tau[j]);
                plds.push(pld + tau[j + 1..].iter().sum::<f64>());
            }
        }
        let s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": plds,
            "BackgroundSuppression": false, "M0Type": "Included", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [3.5, 3.5, 5],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05, 0.10], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.016
        });
        let ctx = format!("volume_type\nm0scan\n{}", "deltam\n".repeat(n * cycles));
        (s, ctx)
    }

    fn h8() -> (Value, String) {
        hadamard_input(8, 2, &[0.25; 7], 0.2)
    }

    #[test]
    fn hadamard_protocols() {
        let hov = |extra: &str| overlay(&format!("[hadamard]\norder = 8\n{extra}"));
        let (s, ctx) = h8();
        let p = parse(&s, &ctx, Some(&hov("")), None).unwrap();
        let h = p.hadamard.as_ref().unwrap();
        assert!(p.p6_active());
        assert_eq!((h.order, h.tau_tot, h.pld, h.report_leakage), (8, 1.75, 0.2, (true, Source::Default)));
        assert_eq!(h.cycles.iter().map(|c| c.rows.clone()).collect::<Vec<_>>(), vec![(1..8).collect::<Vec<_>>(), (8..15).collect()]);
        assert_eq!(h.spans[0], (0.0, 0.25));
        assert!((h.spans[6].1 - 1.75).abs() < 1e-12);
        assert_eq!(parse(&s, &ctx, Some(&hov("report_leakage = false\n")), None).unwrap().hadamard.unwrap().report_leakage,
                   (false, Source::Overlay));
        // unequal sub-boli, the PLDs following them
        let (u, uctx) = hadamard_input(4, 1, &[0.5, 0.3, 0.2], 0.4);
        let hu = parse(&u, &uctx, Some(&overlay("[hadamard]\norder = 4\n")), None).unwrap().hadamard.unwrap();
        assert_eq!((hu.tau.clone(), hu.spans[1]), (vec![0.5, 0.3, 0.2], (0.5, 0.8)));

        let err = |s: &Value, ctx: &str, ov: &str| parse(s, ctx, Some(&overlay(ov)), None).unwrap_err();
        assert!(err(&s, &ctx, "[hadamard]\n").contains("order"));
        assert!(err(&s, &ctx, "[hadamard]\norder = 6\n").contains("order 6"));
        // the deltam count per cycle
        assert!(err(&s, &ctx, "[hadamard]\norder = 4\n").contains("cycles"));
        // m0scan inside a cycle
        let mid = ctx.replacen("deltam\ndeltam\ndeltam\n", "deltam\ndeltam\nm0scan\n", 1);
        let mut sm = s.clone();
        let mut ld = sm["LabelingDuration"].as_array().unwrap().clone();
        ld[3] = json!(0.0);
        sm["LabelingDuration"] = json!(ld);
        let mut pl = sm["PostLabelingDelay"].as_array().unwrap().clone();
        pl[3] = json!(0.0);
        sm["PostLabelingDelay"] = json!(pl);
        assert!(err(&sm, &mid, "[hadamard]\norder = 8\n").contains("m0scan rows go only between cycles"));
        // control and label rows
        let cl = ctx.replacen("deltam", "control", 1);
        assert!(err(&s, &cl, "[hadamard]\norder = 8\n").contains("row 1 is control"));
        // the sub-boli are the same in every cycle
        let mut sv = s.clone();
        sv["LabelingDuration"][9] = json!(0.3);
        assert!(err(&sv, &ctx, "[hadamard]\norder = 8\n").contains("LabelingDuration of row 9"));
        // the effective delays
        let mut sp = s.clone();
        sp["PostLabelingDelay"][2] = json!(1.0);
        let e = err(&sp, &ctx, "[hadamard]\norder = 8\n");
        assert!(e.contains("row 2") && e.contains("effective delay"), "{e}");
        let mut sp = s.clone();
        sp["PostLabelingDelay"][2] = json!(1.45 + 5e-7);
        parse(&sp, &ctx, Some(&hov("")), None).unwrap();
        // one repetition time per cycle
        let mut st = s.clone();
        let mut tr = vec![json!(4.0); 15];
        tr[5] = json!(4.5);
        st["RepetitionTimePreparation"] = json!(tr);
        assert!(err(&st, &ctx, "[hadamard]\norder = 8\n").contains("one repetition per cycle"));
        // PASL
        let mut pa = s.clone();
        pa["ArterialSpinLabelingType"] = json!("PASL");
        pa["BolusCutOffFlag"] = json!(true);
        pa["BolusCutOffTechnique"] = json!("Q2TIPS");
        pa["BolusCutOffDelayTime"] = json!(0.1);
        pa.as_object_mut().unwrap().remove("LabelingDuration");
        assert!(err(&pa, &ctx, "[hadamard]\norder = 8\n").contains("PASL"));
        // compat
        let mut sc = s.clone();
        sc["SliceTiming"] = json!([0.0, 0.0, 0.0]);
        assert!(err(&sc, &ctx, "[hadamard]\norder = 8\n[compat]\nasldro = true\n").contains("compat"));
        // multi-TE and 3D are allowed
        let mut s2 = s.clone();
        s2["EchoTime"] = json!(0.040);
        assert!(parse_echoes(&[s.clone(), s2], &ctx, Some(&hov("")), None).unwrap().hadamard.is_some());
        let mut g = grase();
        let (ld, pl) = (s["LabelingDuration"].clone(), s["PostLabelingDelay"].clone());
        g["LabelingDuration"] = ld;
        g["PostLabelingDelay"] = pl;
        g["M0Type"] = json!("Included");
        let pg = parse(&g, &ctx, Some(&hov("")), None).unwrap();
        assert!(pg.hadamard.is_some() && pg.readout.is_some());
    }

    #[test]
    fn hadamard_suppression_and_crushing_are_per_cycle() {
        let (mut s, ctx) = h8();
        s["BackgroundSuppression"] = json!(true);
        s["BackgroundSuppressionNumberPulses"] = json!(2);
        // the readout is at tau_tot + PLD = 1.95 s from the start of the labeling
        s["BackgroundSuppressionPulseTime"] = json!([1.8, 1.9]);
        let ov = |extra: &str| overlay(&format!("[hadamard]\norder = 8\n{extra}"));
        // BIDS's first-PLD times for every row: one set per cycle (the decoded rows' own delays,
        // 0.2 to 1.7 s, are not the pulses' clock)
        parse(&s, &ctx, Some(&ov("")), None).unwrap();
        s["BackgroundSuppressionPulseTime"] = json!([1.8, 1.95]);
        assert!(parse(&s, &ctx, Some(&ov("")), None).unwrap_err().contains("at or after the readout at 1.95"));
        s["BackgroundSuppressionPulseTime"] = json!([1.8, 1.9]);
        // a set per PLD that differs within a cycle
        let sets: Vec<String> = (0..7).map(|j| format!("[{}, 1.9]", 1.8 + 0.01 * j as f64)).collect();
        let e = parse(&s, &ctx, Some(&ov(&format!("[background_suppression]\npulse_times_per_pld = [{}]\n", sets.join(", ")))), None)
            .unwrap_err();
        assert!(e.contains("cycle 1") && e.contains("pulse sets"), "{e}");
        // a global pulse inside the encoded labeling (after the shortest sub-bolus's end)
        s["BackgroundSuppressionPulseTime"] = json!([1.0, 1.9]);
        let e = parse(&s, &ctx, Some(&ov("[background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n")), None);
        assert!(e.is_ok(), "{e:?}");
        s["BackgroundSuppressionPulseTime"] = json!([1.5, 1.9]);
        let e = parse(&s, &ctx, Some(&ov("")), None).unwrap_err();
        assert!(e.contains("inside the encoded labeling"), "{e}");
        // VENC: one per cycle
        let (mut s, ctx) = h8();
        s["VascularCrushing"] = json!(true);
        s["VascularCrushingVENC"] = json!(4.0);
        let crush = "[vascular_crushing]\nno_arterial_compartment = true\n";
        parse(&s, &ctx, Some(&ov(crush)), None).unwrap();
        let mut venc = vec![0.0];
        venc.extend(vec![4.0; 7]);
        venc.extend(vec![2.0; 7]);
        s["VascularCrushingVENC"] = json!(venc);
        parse(&s, &ctx, Some(&ov(crush)), None).unwrap();
        venc[10] = 3.0;
        s["VascularCrushingVENC"] = json!(venc);
        assert!(parse(&s, &ctx, Some(&ov(crush)), None).unwrap_err().contains("within cycle 2"));
    }

    // ---- P6 part B: Look-Locker protocols

    /// A Look-Locker PCASL sidecar: `m` readouts 0.3 s apart from PLD 0.5 s per cycle, control
    /// then label cycles, gradient echo at 35 degrees (or the per-volume `flips`).
    fn ll_input(m: usize, flips: Option<Vec<f64>>) -> (Value, String) {
        let mut pld = Vec::new();
        let mut ctx = String::from("volume_type\n");
        for kind in ["control", "label"] {
            for n in 0..m {
                pld.push(0.5 + 0.3 * n as f64);
                ctx.push_str(kind);
                ctx.push('\n');
            }
        }
        let s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.0, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 5.0, "LookLocker": true,
            "EchoTime": 0.012, "FlipAngle": flips.map_or(json!(35), |f| json!(f)), "MagneticFieldStrength": 3,
            "AcquisitionVoxelSize": [3.5, 3.5, 5], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05, 0.10],
            "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.016
        });
        (s, ctx)
    }

    const GE: &str = "[signal]\nacq_contrast = \"ge\"\n";

    #[test]
    fn look_locker_protocols() {
        let (s, ctx) = ll_input(4, None);
        let ok = |s: &Value, ctx: &str, ov: &str| parse(s, ctx, Some(&overlay(&format!("{GE}{ov}"))), None);
        let p = ok(&s, &ctx, "").unwrap();
        let ll = p.look_locker.as_ref().unwrap();
        assert!(p.p6_active());
        assert_eq!(ll.cycles.iter().map(|c| c.rows.clone()).collect::<Vec<_>>(), vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        assert_eq!((ll.flip_deg.clone(), ll.flip_array), (vec![35.0; 8], false));
        // the overlay's count is checked against the grouping
        assert_eq!(ok(&s, &ctx, "[look_locker]\nreadouts_per_cycle = 4\n").unwrap().look_locker.unwrap().readouts_per_cycle, Some(4));
        assert!(ok(&s, &ctx, "[look_locker]\nreadouts_per_cycle = 3\n").unwrap_err().contains("readouts_per_cycle = 3"));
        // a non-increasing delay starts a cycle; so do a type change, a TR change and a tau change
        let mut s2 = s.clone();
        s2["PostLabelingDelay"] = json!([0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 1.1, 1.4]);
        let cy: Vec<Vec<usize>> = ok(&s2, &ctx, "").unwrap().look_locker.unwrap().cycles.iter().map(|c| c.rows.clone()).collect();
        assert_eq!(cy, vec![vec![0, 1], vec![2, 3], vec![4, 5, 6, 7]]);
        let mut s3 = s.clone();
        s3["RepetitionTimePreparation"] = json!([5.0, 5.0, 5.5, 5.5, 5.0, 5.0, 5.0, 5.0]);
        assert_eq!(ok(&s3, &ctx, "").unwrap().look_locker.unwrap().cycles.len(), 3);
        // an m0scan row is its own cycle
        let mut s4 = s.clone();
        s4["M0Type"] = json!("Included");
        s4["PostLabelingDelay"] = json!([0.0, 0.5, 0.8, 1.1, 1.4, 0.5, 0.8, 1.1, 1.4]);
        let ctx4 = ctx.replacen("volume_type\n", "volume_type\nm0scan\n", 1);
        let ll4 = ok(&s4, &ctx4, "").unwrap().look_locker.unwrap();
        assert_eq!((ll4.cycles.len(), ll4.cycles[0].m0scan, ll4.cycles[0].rows.clone()), (3, true, vec![0]));

        // flips: per volume in (0, 90]; an array only with LookLocker; not with the overlay's
        let fl: Vec<f64> = (0..8).map(|n| 20.0 + 5.0 * n as f64).collect();
        let (sa, _) = ll_input(4, Some(fl.clone()));
        let lla = ok(&sa, &ctx, "").unwrap().look_locker.unwrap();
        assert_eq!((lla.flip_deg, lla.flip_array), (fl.clone(), true));
        let mut bad = sa.clone();
        bad["FlipAngle"][3] = json!(95);
        assert!(ok(&bad, &ctx, "").unwrap_err().contains("(0, 90]"));
        assert!(ok(&sa, &ctx, "excitation_flip_angle = 30\n").unwrap_err().contains("FlipAngle array"));
        let mut nol = sa.clone();
        nol.as_object_mut().unwrap().remove("LookLocker");
        assert!(ok(&nol, &ctx, "").unwrap_err().contains("only a Look-Locker series"));

        // the separate M0's flip
        let mut sep = sa.clone();
        sep["M0Type"] = json!("Separate");
        assert!(ok(&sep, &ctx, "[m0]\nrepetition_time = 6.0\n").unwrap_err().contains("m0.flip_angle"));
        let m0 = ok(&sep, &ctx, "[m0]\nrepetition_time = 6.0\nflip_angle = 25.0\n").unwrap().look_locker.unwrap().m0_flip_deg;
        assert_eq!(m0, Some((25.0, Source::Overlay)));
        let mut sep1 = s.clone();
        sep1["M0Type"] = json!("Separate");
        assert_eq!(ok(&sep1, &ctx, "[m0]\nrepetition_time = 6.0\n").unwrap().look_locker.unwrap().m0_flip_deg, Some((35.0, Source::Sidecar)));
        assert!(ok(&s, &ctx, "[m0]\nflip_angle = 25.0\n").unwrap_err().contains("separate M0"));
        let mut plain = s.clone();
        plain.as_object_mut().unwrap().remove("LookLocker");
        assert!(ok(&plain, &ctx, "[m0]\nflip_angle = 25.0\n").unwrap_err().contains("Look-Locker"));
        assert!(ok(&plain, &ctx, "[look_locker]\nreadouts_per_cycle = 4\n").unwrap_err().contains("LookLocker: true"));
    }

    #[test]
    fn look_locker_refusals() {
        let (s, ctx) = ll_input(4, None);
        let err = |s: &Value, ov: &str| parse(s, &ctx, Some(&overlay(ov)), None).unwrap_err();
        assert!(err(&s, "").contains("acq_contrast = \"ge\""));
        let mut b = s.clone();
        b["BackgroundSuppression"] = json!(true);
        b["BackgroundSuppressionNumberPulses"] = json!(1);
        b["BackgroundSuppressionPulseTime"] = json!([1.2]);
        // one pulse set per cycle: sets per PLD that differ within a cycle
        let e = err(&b, &format!("{GE}[background_suppression]\npulse_times_per_pld = [[1.2], [1.25], [1.3], [1.35]]\n"));
        assert!(e.contains("one preparation"), "{e}");
        let mut s2 = s.clone();
        s2["EchoTime"] = json!(0.030);
        // P7 part B: Look-Locker with several echoes per readout is accepted
        let multi = parse_echoes(&[s.clone(), s2], &ctx, Some(&overlay(GE)), None).unwrap();
        assert!(multi.look_locker.is_some() && multi.echo_times_s.len() == 2);
        let mut c3 = s.clone();
        c3["SliceTiming"] = json!([0.0, 0.0, 0.0]);
        assert!(err(&c3, &format!("{GE}[compat]\nasldro = true\n")).contains("compat"));
        let mut g = grase();
        g["LookLocker"] = json!(true);
        assert!(parse(&g, CTX, Some(&overlay("")), None).unwrap_err().contains("LookLocker"));
    }

    const MACRO: &str = "[macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.01, csf = 0.0 }\n\
                         arterial_transit_time = { grey_matter = 1.0, white_matter = 1.2, csf = 0.0 }\n";
    const VEL: &str = "[vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n";

    /// P7 part A: the P4 parts are read under Look-Locker as they are without it; crushing's VENC
    /// is per readout; P4's own checks still fire; compat is still refused with each.
    #[test]
    fn look_locker_takes_the_p4_parts() {
        use crate::schedule::Schedule;
        let (s, ctx) = ll_input(2, None);
        let ok = |s: &Value, ov: &str| parse(s, &ctx, Some(&overlay(&format!("{GE}{ov}"))), None);
        // exchange, the arterial compartment
        let p = ok(&s, "[kinetic]\nexchange_time = 0.5\n").unwrap();
        assert!(p.look_locker.is_some() && p.exchange_time.is_some());
        assert!(ok(&s, MACRO).unwrap().macrovascular.is_some());
        // crushing: a VENC alternation over the readouts of two cycles maps readout by readout,
        // and one that varies within a cycle is accepted
        for venc in [json!([0.0, 4.0, 0.0, 4.0]), json!([0.0, 0.0, 4.0, 4.0])] {
            let mut c = s.clone();
            c["VascularCrushing"] = json!(true);
            c["VascularCrushingVENC"] = venc.clone();
            let p = ok(&c, &format!("{MACRO}{VEL}")).unwrap();
            assert_eq!(p.look_locker.as_ref().unwrap().cycles.len(), 2);
            let sch = Schedule::new(&p);
            let got: Vec<Option<f64>> = sch.raws.iter().map(|r| r.venc_with(&sch.preps)).collect();
            let want: Vec<Option<f64>> = venc.as_array().unwrap().iter().map(|v| v.as_f64()).collect();
            assert_eq!(got, want);
        }
        // bolus-position, a slab pulse during the labeling
        let mut b = s.clone();
        b["BackgroundSuppression"] = json!(true);
        b["BackgroundSuppressionNumberPulses"] = json!(1);
        b["BackgroundSuppressionPulseTime"] = json!([0.4]);
        let bp = "[background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n";
        assert!(ok(&b, bp).unwrap().suppression.is_some());
        // P4's own checks under Look-Locker: the VENC floor; the global (P)CASL partial-bolus pulse
        let mut lowv = s.clone();
        lowv["VascularCrushing"] = json!(true);
        lowv["VascularCrushingVENC"] = json!(0.05);
        assert!(ok(&lowv, &format!("{MACRO}{VEL}")).unwrap_err().contains("0.1"));
        let global = "[background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"global\"\n";
        assert!(ok(&b, global).unwrap_err().contains("not yet labeled"));
        // compat with Look-Locker and each P4 part is refused, naming one of them
        let mut c3 = s.clone();
        c3["SliceTiming"] = json!([0.0, 0.0, 0.0]);
        for part in ["[kinetic]\nexchange_time = 0.5\n", MACRO] {
            let e = ok(&c3, &format!("[compat]\nasldro = true\n{part}")).unwrap_err();
            assert!(e.contains("compat") || e.contains("LookLocker"), "{e}");
        }
        // without Look-Locker the raw volumes carry no VENC of their own
        let mut plain = s.clone();
        plain.as_object_mut().unwrap().remove("LookLocker");
        plain["PostLabelingDelay"] = json!(1.5);
        plain["VascularCrushing"] = json!(true);
        plain["VascularCrushingVENC"] = json!([0.0, 4.0, 0.0, 4.0]);
        let p = ok(&plain, &format!("{MACRO}{VEL}")).unwrap();
        assert!(Schedule::new(&p).raws.iter().all(|r| r.venc.is_none()));
    }

    /// A Look-Locker Hadamard-4 PCASL protocol: `cycles` encoding cycles of three 0.3 s sub-boli
    /// read at `PLD_n` = 1.0, 1.3, 1.6 s, the context the decoded volumes readout-major; an m0scan
    /// row before with `m0`.
    fn ll_hadamard(cycles: usize, m0: bool) -> (Value, String) {
        let (h, plds, tau) = (4, [1.0, 1.3, 1.6], 0.3);
        let (mut ld, mut pld, mut ctx) = (Vec::new(), Vec::new(), String::from("volume_type\n"));
        if m0 {
            ld.push(0.0);
            pld.push(0.0);
            ctx.push_str("m0scan\n");
        }
        for _ in 0..cycles {
            for p_n in plds {
                for j in 0..h - 1 {
                    ld.push(tau);
                    pld.push(p_n + tau * (h - 2 - j) as f64);
                    ctx.push_str("deltam\n");
                }
            }
        }
        let s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": if m0 { "Included" } else { "Absent" }, "RepetitionTimePreparation": 5.0,
            "LookLocker": true, "EchoTime": 0.012, "FlipAngle": 35, "MagneticFieldStrength": 3,
            "AcquisitionVoxelSize": [3.5, 3.5, 5], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05, 0.10],
            "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.016
        });
        (s, ctx)
    }

    const LLH: &str = "[hadamard]\norder = 4\n[look_locker]\nreadouts_per_cycle = 3\n";

    /// P7 part B: the Look-Locker Hadamard context, its refusals, and its schedule.
    #[test]
    fn look_locker_hadamard_protocols() {
        use crate::schedule::{Output, Schedule};
        let ok = |s: &Value, ctx: &str, ov: &str| parse(s, ctx, Some(&overlay(&format!("{GE}{ov}"))), None);
        let (s, ctx) = ll_hadamard(2, true);
        let p = ok(&s, &ctx, LLH).unwrap();
        let h = p.hadamard.as_ref().unwrap();
        assert_eq!((h.readouts, h.plds.clone(), h.pld, h.cycles.len()), (3, vec![1.0, 1.3, 1.6], 1.0, 2));
        assert_eq!(h.cycles[0].rows, (1..10).collect::<Vec<_>>());
        assert_eq!(h.row(&h.cycles[1], 2, 1), 10 + 3 + 2);
        // the delays decrease within a readout's block, so the run rule would have split every row;
        // the Look-Locker cycles come from the encoding: sub-bolus 1 of each readout
        let ll = p.look_locker.as_ref().unwrap();
        assert_eq!(ll.cycles.iter().map(|c| (c.rows.clone(), c.m0scan)).collect::<Vec<_>>(),
                   vec![(vec![0], true), (vec![1, 4, 7], false), (vec![10, 13, 16], false)]);
        assert!(ll.cycles[1].rows.iter().map(|&r| p.rows[r].t).collect::<Vec<_>>()
            .iter().zip([1.9, 2.2, 2.5]).all(|(a, b)| (a - b).abs() < 1e-12));

        // the schedule: per encoding row one preparation and three raw volumes, outputs readout-major
        let sch = Schedule::new(&p);
        assert_eq!((sch.preps.len(), sch.raws.len(), sch.outputs.len()), (1 + 2 * 4, 1 + 2 * 4 * 3, 1 + 2 * 9));
        for e in 0..4 {
            for n in 0..3 {
                let r = &sch.raws[1 + e * 3 + n];
                assert_eq!((r.prep, r.readout, r.encoding_row, r.cycle), (1 + e, n, Some(e), Some(0)));
                assert!((sch.raw_rows[1 + e * 3 + n].t - (0.9 + [1.0, 1.3, 1.6][n])).abs() < 1e-12);
            }
            assert_eq!(sch.preps[1 + e].raw, 1 + e * 3);
        }
        assert_eq!(sch.outputs[1..10].to_vec(), (0..3).flat_map(|n| (0..3).map(move |j| Output::Decoded { cycle: 0, subbolus: j, readout: n }))
            .collect::<Vec<_>>());
        assert_eq!(sch.cycles[1].raws, 13..25);
        assert!((sch.preps[1 + 4].start_s - (5.0 + 4.0 * 5.0)).abs() < 1e-12, "{}", sch.preps[5].start_s);

        // refusals
        let err = |s: &Value, ctx: &str, ov: &str| ok(s, ctx, ov).unwrap_err();
        let (s1, ctx1) = ll_hadamard(1, false);
        assert!(err(&s1, &ctx1, "[hadamard]\norder = 4\n").contains("readouts_per_cycle"));
        assert!(err(&s1, &ctx1, "[hadamard]\norder = 4\n[look_locker]\nreadouts_per_cycle = 2\n").contains("9 deltam rows"));
        let lab = ctx1.replacen("deltam", "label", 1);
        assert!(err(&s1, &lab, LLH).contains("is label"));
        let mut d = s1.clone();
        d["PostLabelingDelay"][4] = json!(1.55);
        let e = err(&d, &ctx1, LLH);
        assert!(e.contains("sub-bolus 2, readout 2"), "{e}");
        // readout 2 earlier than readout 1
        let mut dec = s1.clone();
        for j in 0..3 {
            dec["PostLabelingDelay"][3 + j] = json!(0.9 + 0.3 * (2 - j) as f64);
        }
        assert!(err(&dec, &ctx1, LLH).contains("strictly increase"));
        let mut fl = s1.clone();
        fl["FlipAngle"] = json!([35, 35, 35, 30, 35, 35, 25, 25, 25]);
        let e = err(&fl, &ctx1, LLH);
        assert!(e.contains("FlipAngle") && e.contains("sub-bolus 2, readout 2"), "{e}");
        let mut flok = s1.clone();
        flok["FlipAngle"] = json!([35, 35, 35, 30, 30, 30, 25, 25, 25]);
        ok(&flok, &ctx1, LLH).unwrap();
        // one readout per cycle: its sub-boli still share the excitation (final review 1)
        let mut one = s1.clone();
        for (_, v) in one.as_object_mut().unwrap().iter_mut() {
            if let Value::Array(a) = v {
                if a.len() == 9 {
                    a.truncate(3);
                }
            }
        }
        let ctx_one = "volume_type\ndeltam\ndeltam\ndeltam\n";
        let one_ov = "[hadamard]\norder = 4\n[look_locker]\nreadouts_per_cycle = 1\n";
        one["FlipAngle"] = json!([35, 35, 35]);
        ok(&one, ctx_one, one_ov).unwrap();
        one["FlipAngle"] = json!([35, 30, 35]);
        let e = err(&one, ctx_one, one_ov);
        assert!(e.contains("FlipAngle") && e.contains("row 1") && e.contains("sub-bolus 2"), "{e}");
        // crushing: one VENC per readout, the same for its sub-boli
        let crush = format!("{LLH}{MACRO}{VEL}");
        let mut cr = s1.clone();
        cr["VascularCrushing"] = json!(true);
        cr["VascularCrushingVENC"] = json!([0, 0, 0, 4, 4, 4, 0, 0, 0]);
        let pc = ok(&cr, &ctx1, &crush).unwrap();
        let sc = Schedule::new(&pc);
        assert_eq!(sc.raws[..3].iter().map(|r| r.venc_with(&sc.preps)).collect::<Vec<_>>(), vec![Some(0.0), Some(4.0), Some(0.0)]);
        let mut crb = cr.clone();
        crb["VascularCrushingVENC"] = json!([0, 0, 0, 4, 0, 4, 0, 0, 0]);
        assert!(err(&crb, &ctx1, &crush).contains("VascularCrushingVENC of row 4"));
        // without Look-Locker P6's rule stands: three P6 cycles (their own effective delays), a VENC
        // that varies within one is refused, one per cycle is accepted
        let mut plain = cr.clone();
        plain.as_object_mut().unwrap().remove("LookLocker");
        plain["PostLabelingDelay"] = json!([1.6, 1.3, 1.0, 1.6, 1.3, 1.0, 1.6, 1.3, 1.0]);
        let ov_plain = overlay(&format!("{GE}[hadamard]\norder = 4\n{MACRO}{VEL}"));
        parse(&plain, &ctx1, Some(&ov_plain), None).unwrap();
        plain["VascularCrushingVENC"] = json!([0, 4, 0, 0, 4, 0, 0, 4, 0]);
        let e = parse(&plain, &ctx1, Some(&ov_plain), None).unwrap_err();
        assert!(e.contains("one VENC per cycle"), "{e}");
        // suppression: one set per cycle, its pulses before the first readout (1.9 s)
        let mut bs = s1.clone();
        bs["BackgroundSuppression"] = json!(true);
        bs["BackgroundSuppressionNumberPulses"] = json!(1);
        bs["BackgroundSuppressionPulseTime"] = json!([1.5]);
        ok(&bs, &ctx1, LLH).unwrap();
        bs["BackgroundSuppressionPulseTime"] = json!([2.0]);
        assert!(err(&bs, &ctx1, LLH).contains("at or after the readout"));
    }

    #[test]
    fn look_locker_timing_and_schedule() {
        use crate::schedule::{Output, Schedule};
        let dims = [32, 32, 3];
        let run = |s: &Value, ctx: &str| check_excitation_timing(&parse(s, ctx, Some(&overlay(GE)), None).unwrap(), dims);
        // h = 8 ms, TE 12 ms: a slice group reads until its excitation + 20 ms
        let (s, ctx) = ll_input(4, None);
        run(&s, &ctx).unwrap();
        // slices 15 ms apart: a group's block overruns the next group's excitation
        let mut a = s.clone();
        a["SliceTiming"] = json!([0.0, 0.015, 0.03]);
        assert!(run(&a, &ctx).unwrap_err().contains("next slices' excitation"));
        // interleaved and reversed orders are the same groups
        for t in [json!([0.0, 0.10, 0.05]), json!([0.10, 0.05, 0.0])] {
            let mut b = s.clone();
            b["SliceTiming"] = t;
            run(&b, &ctx).unwrap();
        }
        // multiband: two groups
        let mut mb = s.clone();
        mb["SliceTiming"] = json!([0.0, 0.05, 0.0, 0.05]);
        mb["MultibandAccelerationFactor"] = json!(2);
        mb["AcquisitionVoxelSize"] = json!([3.5, 3.5, 5]);
        run(&mb, &ctx).unwrap();
        // readouts 0.1 s apart: the last group (0.10 s) reads past the next readout
        let mut c = s.clone();
        c["PostLabelingDelay"] = json!([0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8]);
        assert!(run(&c, &ctx).unwrap_err().contains("next readout"));
        // the last readout before TR: t = 1.0 + 1.4 = 2.4, its last slices read until 2.52 s
        let mut d = s.clone();
        d["RepetitionTimePreparation"] = json!(2.51);
        assert!(run(&d, &ctx).unwrap_err().contains("RepetitionTimePreparation"));

        // P7 part B: echoes at 12 and 30 ms (blocks of +-8 ms; the second ends 38 ms after each
        // slice group's excitation, inside the 50 ms between groups). Readouts 0.13 s apart fit one
        // echo (the last slices, at 0.10 s, read until 0.12 s) but not two (until 0.138 s): the
        // second echo of a readout overlaps the next readout, and the error names both
        let echoes = |base: &Value, tes: &[f64]| -> Vec<Value> {
            tes.iter().map(|te| { let mut x = base.clone(); x["EchoTime"] = json!(te); x }).collect()
        };
        let run_e = |side: &[Value]| check_excitation_timing(&parse_echoes(side, &ctx, Some(&overlay(GE)), None).unwrap(), dims);
        let mut tight = s.clone();
        tight["PostLabelingDelay"] = json!([0.5, 0.63, 0.76, 0.89, 0.5, 0.63, 0.76, 0.89]);
        run_e(&echoes(&tight, &[0.012])).unwrap();
        let e2 = run_e(&echoes(&tight, &[0.012, 0.030])).unwrap_err();
        assert!(e2.contains("echo 2") && e2.contains("next readout"), "{e2}");
        run_e(&echoes(&s, &[0.012, 0.030])).unwrap();
        // a third echo at 50 ms overruns the next slice group's excitation
        let g = run_e(&echoes(&s, &[0.012, 0.030, 0.050])).unwrap_err();
        assert!(g.contains("echo 3") && g.contains("next slices"), "{g}");
        // echoes whose blocks overlap within an excitation
        let ov = run_e(&echoes(&s, &[0.012, 0.025])).unwrap_err();
        assert!(ov.contains("overlap"), "{ov}");

        // the schedule: one preparation per cycle, its readouts the raw volumes, one TR per cycle
        let p = parse(&s, &ctx, Some(&overlay(GE)), None).unwrap();
        let sch = Schedule::new(&p);
        assert_eq!((sch.preps.len(), sch.raws.len(), sch.outputs.len(), sch.cycles.len()), (2, 8, 8, 2));
        assert_eq!((sch.preps[1].start_s, sch.preps[0].labeling_window), (5.0, [0.0, 1.0]));
        for (r, raw) in sch.raws.iter().enumerate() {
            assert_eq!((raw.prep, raw.readout, raw.cycle, raw.n_preps), (r / 4, r % 4, Some(r / 4), 1));
            assert_eq!(sch.outputs[r], Output::Raw(r));
            assert_eq!(sch.raw_rows[r], p.rows[r]);
        }
    }

    /// Review fixes: under [hadamard] motion volumes index the raw volumes (8 for one H8 cycle,
    /// 7 decoded rows); Look-Locker refuses a missing FlipAngle.
    #[test]
    fn hadamard_motion_and_look_locker_flip_rules() {
        let (s, ctx) = hadamard_input(8, 1, &[0.25; 7], 0.2);
        let ov = |v: &str| overlay(&format!(
            "[hadamard]\norder = 8\n[motion]\nmode = \"random\"\ntrans_mm = [1.0, 0.0, 0.0]\nrot_deg = [0.0, 0.0, 0.0]\nvolumes = [{v}]\n"));
        // an m0scan row and eight raw volumes: indices 0..9
        parse(&s, &ctx, Some(&ov("0, 8")), None).unwrap();
        assert!(parse(&s, &ctx, Some(&ov("9")), None).unwrap_err().contains("outside the 9 volumes"));
        let (mut l, lctx) = ll_input(4, None);
        l.as_object_mut().unwrap().remove("FlipAngle");
        assert!(parse(&l, &lctx, Some(&overlay(GE)), None).unwrap_err().contains("requires FlipAngle"));
        // the overlay's excitation angle gives it as well
        parse(&l, &lctx, Some(&overlay(&format!("{GE}excitation_flip_angle = 30\n"))), None).unwrap();
    }
}
