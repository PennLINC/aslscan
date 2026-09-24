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

use std::path::Path;

use mrsim_acq::kspace::{Acquisition, KspaceWindow, PartialFourierMode};
use serde::Deserialize;
use serde_json::Value;

use crate::kinetic::{Kinetic, LabelType};
use crate::mrsignal::{parse_contrast, Contrast};
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
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Sidecar => "Sidecar",
            Source::Phantom => "Phantom",
            Source::Overlay => "Overlay",
            Source::Default => "Default",
        }
    }
}

/// The `phantom.json` block a converted phantom carries; the kinetic constants the oracle used.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhantomParams {
    pub lambda: Option<f64>,
    pub t1b: Option<f64>,
    pub field_strength: Option<f64>,
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
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KineticOverlay {
    pub label_efficiency: Option<f64>,
    pub lambda_blood_brain: Option<f64>,
    pub t1_arterial_blood: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalOverlay {
    pub acq_contrast: Option<String>,
    /// s
    pub t2_blood: Option<f64>,
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
}

/// The resolved protocol. Seconds unless the field name says `_ms`.
#[derive(Debug, Clone)]
pub struct Protocol {
    pub label_type: LabelType,
    pub rows: Vec<Row>,
    pub m0_type: M0Type,
    pub background_suppression: bool,
    /// Per acquired slice in data z order (after `SliceEncodingDirection`), minus the minimum (s).
    pub slice_offsets: Vec<f64>,
    pub field_strength: f64,
    pub voxel_size_mm: [f64; 3],
    pub reverse_phase: bool,
    pub phase_encoding_direction: String,
    pub echo_time_s: f64,
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

// ---------------------------------------------------------------- the parse

/// Parse from already-read text. [`load`] is the file-reading wrapper.
pub fn parse(sidecar: &Value, aslcontext: &str, overlay: Option<&Overlay>, phantom: Option<&PhantomParams>)
    -> Result<Protocol, String>
{
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

    // The readout's dimensionality is the dominant P1 limitation, so it is reported before any
    // per-row timing problem a 3D dataset might also have.
    let mr_type = string(sidecar, "MRAcquisitionType")?;
    if mr_type != "2D" {
        return Err(format!(
            "asl.json: MRAcquisitionType {mr_type:?}: P1 simulates 2D EPI only; 3D readouts arrive with P5"));
    }

    // Things P1 does not model must not be silently simulated as if absent.
    if opt_bool(sidecar, "LookLocker")? {
        return Err("asl.json: LookLocker is true; Look-Locker readouts arrive with P6".to_string());
    }
    if opt_bool(sidecar, "VascularCrushing")? {
        return Err("asl.json: VascularCrushing is true; vascular crushing arrives with P4".to_string());
    }
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

    // Readout geometry and timing.
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

    let ped = string(sidecar, "PhaseEncodingDirection")?;
    let reverse_phase = match ped.as_str() {
        "j-" => false,
        "j" => true,
        _ => {
            return Err(format!(
                "asl.json: PhaseEncodingDirection {ped:?}: P1 encodes phase along the second data axis \
                 (j or j-) only"))
        }
    };
    let (te, _) = num_or_array(sidecar, "EchoTime")?;
    if te.iter().any(|v| *v != te[0]) {
        return Err(format!(
            "asl.json: EchoTime array has unequal entries {te:?}; multi-TE ASL arrives with P6"));
    }
    let echo_time_s = require_finite_positive(te[0], "asl.json: EchoTime")?;
    let total_readout_time_s = require_finite_positive(num(sidecar, "TotalReadoutTime")?, "asl.json: TotalReadoutTime")?;
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

    let m0_repetition_time_s = overlay.and_then(|o| o.m0.as_ref()).and_then(|m| m.repetition_time);
    if let Some(v) = m0_repetition_time_s {
        require_finite_positive(v, "overlay m0.repetition_time")?;
    }
    if m0_type == M0Type::Separate && m0_repetition_time_s.is_none() {
        return Err("M0Type is \"Separate\" but the overlay has no [m0] repetition_time; the ASL sidecar's \
                    RepetitionTimePreparation describes the ASL series, not the M0 scan".to_string());
    }
    let acq = overlay_acq(overlay.and_then(|o| o.acquisition.as_ref()))?;
    let seed = overlay.and_then(|o| o.seed).unwrap_or(0);

    Ok(Protocol {
        label_type, rows, m0_type, background_suppression, slice_offsets, field_strength,
        voxel_size_mm, reverse_phase, phase_encoding_direction: ped, echo_time_s,
        total_readout_time_s, accel, mb, alpha, lambda, t1b, t2_blood_s, contrast,
        m0_repetition_time_s, seed, acq, input_sidecar: sidecar.clone(),
    })
}

/// Read the files and [`parse`].
pub fn load(asl_json: &Path, aslcontext_tsv: &Path, overlay: Option<&Path>, phantom: Option<&PhantomParams>)
    -> Result<Protocol, String>
{
    let sidecar: Value = serde_json::from_str(
        &std::fs::read_to_string(asl_json).map_err(|e| format!("{}: {e}", asl_json.display()))?,
    )
    .map_err(|e| format!("{}: {e}", asl_json.display()))?;
    let ctx = std::fs::read_to_string(aslcontext_tsv).map_err(|e| format!("{}: {e}", aslcontext_tsv.display()))?;
    let ov: Option<Overlay> = match overlay {
        None => None,
        Some(p) => Some(
            toml::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?)
                .map_err(|e| format!("{}: {e}", p.display()))?,
        ),
    };
    parse(&sidecar, &ctx, ov.as_ref(), phantom)
}

impl Protocol {
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
            do_relaxation: true,
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
        };
        mrsim_acq::kspace::validate_acquisition_timing(&acq, nx, ny).map_err(|inner| {
            format!(
                "EchoTime {} s with TotalReadoutTime {} s on a {ny}-line readout: the first acquired line \
                 precedes the excitation ({inner}). Raise EchoTime or bring TotalReadoutTime below about \
                 {} s; partial Fourier and in-plane acceleration do not shorten the pre-echo readout in \
                 this model.",
                self.echo_time_s, self.total_readout_time_s, 2.0 * self.echo_time_s)
        })?;
        Ok(acq)
    }
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
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("P6"));
        let mut s = base();
        s["VascularCrushing"] = json!(true);
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("P4"));
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
        let mut s = base();
        s["MRAcquisitionType"] = json!("3D");
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("P5"));
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
        assert!(parse(&s, CTX, Some(&m0_overlay()), None).unwrap_err().contains("P6"));
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
        let ph = PhantomParams { lambda: Some(0.91), t1b: Some(1.7), field_strength: Some(3.0) };
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
        // ge / ir rejected through the overlay
        let ov = overlay("[signal]\nacq_contrast = \"ge\"\n[m0]\nrepetition_time = 8.0\n");
        assert!(parse(&base(), CTX, Some(&ov), None).unwrap_err().contains("P5"));
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
        // asl004: 2D PCASL with a 96-entry PLD array, LabelingEfficiency in the sidecar, and a
        // trailing blank line in aslcontext.tsv.
        let (s, c) = fixture("asl004");
        let p = parse(&s, &c, Some(&m0_overlay()), None).unwrap();
        assert_eq!(p.rows.len(), 96);
        assert_eq!(p.alpha, (0.88, Source::Sidecar));
        assert!((p.rows[0].t - (0.25 + 1.4)).abs() < 1e-12 && (p.rows[95].t - (1.5 + 1.4)).abs() < 1e-12);
        // Its TE 14 ms / TRT 60 ms readout starts before the excitation under this line-timing model.
        let e = p.acquisition(58, 58).unwrap_err();
        assert!(e.contains("EchoTime"), "{e}");
        // asl001, asl003, asl005 are 3D and are rejected naming P5.
        for ex in ["asl001", "asl003", "asl005"] {
            let (s, c) = fixture(ex);
            let e = parse(&s, &c, Some(&m0_overlay()), None).unwrap_err();
            assert!(e.contains("P5"), "{ex}: {e}");
        }
        // asl003 as a 2D variant: PASL Q2TIPS with a 20-entry PLD array whose first entries
        // (0.3 s) precede the 0.7 s bolus cutoff, which the PASL check must refuse...
        let (mut s, c) = fixture("asl003");
        s["MRAcquisitionType"] = json!("2D");
        s["SliceTiming"] = json!([0.0, 0.04, 0.08]);
        s["TotalReadoutTime"] = json!(0.02);
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
}
