//! BIDS output: names, `_aslcontext.tsv`, the sidecars, `dataset_description.json`,
//! `.bidsignore`, and the ground-truth maps (spec: output contract).
//!
//! The naming and TSV code is pure std so it tests offline; the writer needs `io`.

use crate::rows::Row;

/// Subject and optional session, without their `sub-` / `ses-` prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Names {
    pub sub: String,
    pub ses: Option<String>,
}

impl Names {
    pub fn new(sub: &str, ses: Option<&str>) -> Names {
        Names { sub: sub.trim_start_matches("sub-").to_string(), ses: ses.map(|s| s.trim_start_matches("ses-").to_string()) }
    }

    /// `sub-XX[_ses-YY]`
    pub fn stem(&self) -> String {
        match &self.ses {
            Some(s) => format!("sub-{}_ses-{}", self.sub, s),
            None => format!("sub-{}", self.sub),
        }
    }

    /// `sub-XX[/ses-YY]/perf`, relative to the dataset root.
    pub fn perf_dir(&self) -> String {
        match &self.ses {
            Some(s) => format!("sub-{}/ses-{}/perf", self.sub, s),
            None => format!("sub-{}/perf", self.sub),
        }
    }

    /// `<perf_dir>/<stem><tail>`
    pub fn rel(&self, tail: &str) -> String {
        format!("{}/{}{}", self.perf_dir(), self.stem(), tail)
    }
}

/// The `_aslcontext.tsv` text: a `volume_type` header, one row per volume in series order.
pub fn aslcontext_tsv(rows: &[Row]) -> String {
    let mut s = String::from("volume_type\n");
    for r in rows {
        s.push_str(r.kind.as_str());
        s.push('\n');
    }
    s
}

/// The ground-truth filename tail for a map name.
pub fn ground_truth_tail(desc: &str) -> String {
    format!("_desc-{desc}_gt.nii.gz")
}

#[cfg(feature = "io")]
pub use writer::write_dataset;

#[cfg(feature = "io")]
mod writer {
    use std::path::{Path, PathBuf};

    use mrsim_acq::io::{write_3d, write_3d_i16, write_4d, write_complex_4d, SidecarInfo};
    use serde_json::{json, Map, Value};

    use super::{aslcontext_tsv, ground_truth_tail, Names};
    use crate::phantom::T2Mode;
    use crate::bolus::Region;
    use crate::protocol::{M0Type, Protocol, QuantitySource, SuppressionModel, COMPAT_PINNED};
    use crate::resample::GridOrigin;
    use crate::series::{LookLockerSeries, SeriesOutput};

    fn write_json(path: &Path, v: &Value) -> Result<(), String> {
        let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    fn resolved(v: (f64, crate::protocol::Source)) -> Value {
        json!({ "Value": v.0, "Source": v.1.as_str() })
    }

    /// P6 part C: the multi-TE record of echo `e` (from 0), `None` with one echo.
    fn multi_echo_block(p: &Protocol, out: &SeriesOutput, e: usize) -> Option<Value> {
        let m = p.multi_te.as_ref()?;
        let seed = out.seeds.0;
        let salts: Vec<u64> = (0..p.echo_times_s.len()).map(mrsim_acq::kspace::echo_salt).collect();
        let gradient = p.contrast == crate::mrsignal::Contrast::GradientEcho;
        let decay = match (p.compat.is_some(), gradient) {
            (true, _) => "compat: simasl's exp(-TE/T2) per phantom voxel (exp(-TE/T2*) under gradient echo) in the \
                          signal stage, one image set per echo; the readout applies no relaxation",
            (false, false) => "per compartment, by the readout, on the object before encoding: spin echo \
                               exp(-(TE + t)/T2 - |t|/T2') about each echo, so delta-M(TE) = dM_iv exp(-TE/T2_blood) + \
                               dM_ev exp(-TE/T2_tissue) [+ the arterial term at T2_arterial]; perfect refocusing",
            (false, true) => "per compartment, by the readout, on the object before encoding: gradient echo \
                              exp(-(TE + t)(1/T2 + 1/T2')) exp(i 2 pi fmap TE), so delta-M(TE) = dM_iv \
                              exp(-TE/T2*_blood) + dM_ev exp(-TE/T2*_tissue) [+ the arterial term], with the fieldmap phase",
        };
        let mut b = json!({
            "Echo": e + 1,
            "EchoTimes": p.echo_times_s,
            "EchoFormation": if gradient { "gradient" } else { "spin" },
            "EchoTrain": "each echo an independent 2D EPI readout of the same excitation at its own echo time; the \
                          echo train's own k-space trajectory is not modeled",
            "ExcitationSeed": seed,
            "ReceiverSeeds": salts.iter().map(|s| seed ^ s).collect::<Vec<u64>>(),
            "ReceiverSeedRule": "seed XOR echo_salt(e), echo_salt(e) = e * 0xD1B54A32D192ED03 (mod 2^64), e from 0: \
                                 the echoes share the excitation (one shot phase) and draw independent receiver noise",
            "Decay": decay,
            "Exchange": if p.exchange_time.is_some() {
                "on: the intravascular label in the blood compartment, the extravascular label in the tissue's"
            } else {
                "off: all label is intravascular (blood relaxation)"
            },
            "Kinetics": "P4's single residue T1' after arrival; not a two-compartment T1 exchange model",
            "Timing": if p.compat.is_some() {
                "each echo's readout block, the refocusing pulses and the last echo against TR checked; compat \
                 excites every slice together (equal SliceTiming), so there is no between-group check"
            } else {
                "each echo's readout block, the refocusing pulses, the excitation groups in order and the last \
                 echo against TR checked"
            },
        });
        if let Some(r) = m.refocusing_time_ms {
            b["RefocusingTime"] = resolved(r);
        }
        if let Some(l) = m.max_image_memory_gib {
            b["MaxImageMemoryGiB"] = resolved(l);
        }
        if let Some(m0) = out.seeds.1 {
            b["SeparateM0ReceiverSeeds"] = json!(salts.iter().map(|s| m0 ^ s).collect::<Vec<u64>>());
        }
        Some(b)
    }

    /// The `PulseSequenceType` to publish when the overlay's `[readout] type` changed the readout
    /// the input's `PulseSequenceType` describes (P5 part B): the readout simulated, the input's value
    /// kept under `InputValuesReplaced`.
    fn overridden_sequence_type(p: &Protocol) -> Option<&'static str> {
        let rs = p.readout.as_ref()?;
        let input = p.input_sidecar.get("PulseSequenceType").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
        (rs.kind.1 == crate::protocol::Source::Overlay && !input.contains(rs.kind.0.as_str())).then_some(match rs.kind.0 {
            crate::protocol::ReadoutKind::Grase => "GRASE",
            crate::protocol::ReadoutKind::Spiral => "spiral",
        })
    }

    /// JSON equality with numbers compared as numbers (an input `90` is not replaced by `90.0`),
    /// elementwise through arrays.
    fn same_number(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_number(p, q)),
            _ => match (a.as_f64(), b.as_f64()) {
                (Some(p), Some(q)) => p == q,
                _ => a == b,
            },
        }
    }

    /// The per-row label factors as one number when they agree, else the array.
    fn label_factor_value(f: &[f64]) -> Value {
        match f.first() {
            Some(x) if f.iter().all(|y| y == x) => json!(x),
            _ => json!(f),
        }
    }

    /// Everything the simulator resolved, for the output sidecar.
    fn simulation_block(p: &Protocol, out: &SeriesOutput) -> Value {
        let a = &out.acquisition;
        let suppression = match (&p.suppression, &out.label_factors) {
            (Some(s), Some(f)) => json!({
                "Model": "global-bolus",
                "ModelNote": "every pulse inverts the whole labeled bolus wherever it is; an upper bound on \
                              the retained label (P3 addendum, part A)",
                "InversionEfficiency": resolved(s.epsilon),
                "Presaturation": { "Value": s.presaturation.0, "Source": s.presaturation.1.as_str() },
                "LabelFactor": label_factor_value(f),
                "FirstPldPulseTimesAppliedToAll": s.first_pld_applied_to_all,
                "PulseTimesPerRow": s.per_row,
                "TissueModel": "signed longitudinal timeline per acquired slice; m0scan rows unsuppressed",
            }),
            _ => Value::Null,
        };
        let ir = p.ir.as_ref().map_or(Value::Null, |s| json!({
            "InversionTime": { "Value": s.params.inversion_time, "Source": s.inversion_time.as_str() },
            "ExcitationFlipAngle": { "Value": s.params.excitation_flip_deg, "Source": s.excitation_flip.as_str() },
            "InversionFlipAngle": { "Value": s.params.inversion_flip_deg, "Source": s.inversion_flip.as_str() },
            "BloodModel": "sin(FlipAngle) * delta_m; the preparation does not invert the label (simasl)",
        }));
        let motion = p.motion.as_ref().map_or(Value::Null, |m| json!({
            "Mode": m.mode_name,
            "Seed": out.motion_seed,
            "SeedSalt": format!("{:#x}", crate::series::MOTION_SEED_SALT),
            "RotationOrder": "Rz Ry Rx, degrees, about the field-of-view centre",
            "WithinVolume": m.within.as_ref().map(|w| json!({
                "DropoutRate": w.dropout_rate, "Severity": w.severity,
                "JumpMm": w.jump_mm, "JumpDeg": w.jump_deg, "Events": out.events.len(),
            })),
            "GroundTruth": ["desc-motion_gt.tsv (rotations in radians)", "desc-motionEvents_gt.tsv",
                            "desc-deltam_gt (moved)", "desc-deltamStatic_gt (unmoved)"],
            "Approximations": [
                "finished simulation-grid images are resampled, so slice timing travels with the anatomy",
                "per-voxel T2/T2' maps (voxel mode) and the fieldmap stay in scanner space",
            ],
        }));
        let mut block = json!({
            "Simulator": { "Name": "aslscan", "Version": env!("CARGO_PKG_VERSION") },
            "Seed": out.seeds.0,
            "M0ScanSeed": out.seeds.1,
            "M0ScanContrast": "se",
            "T2Mode": out.mode.as_str(),
            "Labels": out.labels.iter().map(|(l, n)| json!({ "Label": l, "Name": n })).collect::<Vec<_>>(),
            "Compartments": out.n_compartments,
            "CompartmentOrder": "tissue per label, then labeled blood per label",
            "Resolved": {
                "LabelingEfficiency": resolved(p.alpha),
                "LambdaBloodBrain": resolved(p.lambda),
                "T1ArterialBlood": resolved(p.t1b),
                "T2Blood": resolved(p.t2_blood_s),
                "AcqContrast": p.contrast.as_str(),
                "M0RepetitionTime": p.m0_repetition_time_s,
            },
            "BackgroundSuppressionLabelFactor": out.label_factors.as_deref().map(label_factor_value),
            "BackgroundSuppressionModel": p.suppression.as_ref().map(|_| "global-bolus"),
            "BackgroundSuppression": suppression,
            "InversionRecovery": ir,
            "Motion": motion,
            "Grid": {
                "AcquisitionMatrix": out.acq_grid.dims,
                "SimulationMatrix": out.sim_grid.dims,
                "Oversample": p.acq.oversample,
            },
            "Acquisition": {
                "TLineMs": a.t_line, "TEchoMs": a.t_echo, "TInhomFallbackMs": a.t_inhom,
                "SignalScale": a.signal_scale, "ReversePhase": a.reverse_phase,
                "DoDistortions": a.do_distortions, "NoiseVariance": a.noise_variance,
                "PartialFourier": a.partial_fourier, "PfMode": format!("{:?}", a.pf_mode),
                "GhostOffset": a.ghost_offset, "EddyStrength": a.eddy_strength, "EddyQuad": a.eddy_quad,
                "EddyPhase": a.eddy_phase, "EddyTauMs": a.eddy_tau, "NSpikes": a.n_spikes,
                "SpikeAmplitude": a.spike_amplitude, "Window": format!("{:?}", a.window),
                "NCoils": a.n_coils, "Accel": a.accel, "AcsLines": a.acs_lines,
            },
            "FieldmapPresent": out.fieldmap_present,
            "SliceOffsetsS": p.slice_offsets,
        });
        // Written only when used, so a P1/P3 sidecar is unchanged.
        if p.grid_origin != GridOrigin::Corner {
            block["Grid"]["Origin"] = json!(p.grid_origin.as_str());
        }
        p4_blocks(&mut block, p, out);
        // P5 parts B and D: written only for a 3D readout
        if let (Some(r3), Some(rs)) = (&out.readout, &p.readout) {
            let [_, ny, nz] = out.acq_grid.dims;
            let tr0 = p.rows.iter().find(|r| r.kind != crate::rows::RowKind::M0scan).or(p.rows.first()).map_or(0.0, |r| r.tr);
            let mut approx = vec![
                "an ideal slab: exactly the field of view in z, uniform, no kz aliasing",
                "no through-plane oversampling: partitions are the acquisition grid's z cells",
                "the echo train leaves Mz = 0 at the excitation and recovery is counted from it",
                "coil sensitivities uniform in z",
            ];
            if r3.t_line_source.contains("DwellTime") {
                approx.push("the line spacing from DwellTime is a lower bound (no ramps, no receiver oversampling)");
            }
            if out.mode == T2Mode::Voxel && r3.train.refocusing_deg != 180.0 {
                approx.push("voxel mode: a mixed cell's T1 is the M0-weighted rate mean, as for T2 (not a mixture of echo trains)");
            }
            if let Some(sp) = &r3.spiral {
                approx.push("the spiral's sampling bound is on k-space speed (one cycle/FOV per dwell time), not a gradient or slew limit");
                approx.push("an ideal trajectory: no gradient delays or eddy currents");
                let segs = out.spiral_segmentation.as_deref().unwrap_or(&[]);
                let fold = |voxel: bool| {
                    let s: Vec<_> = segs.iter().filter(|g| g.voxel == voxel).collect();
                    (!s.is_empty()).then(|| json!({
                        "Segments": s.iter().map(|g| g.l).max(), "ChebyshevDegree": s.iter().map(|g| g.m).max(),
                        "CoefficientSum": s.iter().map(|g| g.b_sum).fold(0.0, f64::max),
                        "CertifiedBound": s.iter().map(|g| g.bound).fold(0.0, f64::max),
                        "PerSlice": s.iter().map(|g| json!([g.l, g.m, g.bound])).collect::<Vec<_>>(),
                    }))
                };
                block["Readout"] = json!({
                    "Type": rs.kind.0.as_str(), "TypeSource": rs.kind.1.as_str(),
                    "NumberShots": r3.n_shots, "Interleaves": sp.interleaves, "KzSegments": rs.kz_segments.0,
                    "EchoTrainLength": r3.etl,
                    "EchoSpacingMs": r3.esp_ms,
                    "EchoSpacingSource": if rs.echo_spacing_ms.is_some() { "overlay readout.echo_spacing" }
                                         else { "EchoTime, the k-space-centre time: each spiral starts at its echo, so CentreEcho x ESP" },
                    "KzOrder": match rs.kz_order.0 { mrsim_acq::readout::KzOrder::Centric => "centric", _ => "linear" },
                    "CentreEcho": r3.e_c,
                    "RefocusingFlipAngle": { "Value": rs.refocusing_flip_deg.0, "Source": rs.refocusing_flip_deg.1.as_str() },
                    "FlipAngleInterpretation": "the sidecar's FlipAngle is the echo train's refocusing angle; the excitation is 90 degrees",
                    "RefocusingTimeMs": { "Value": rs.refocusing_time_ms.0, "Source": rs.refocusing_time_ms.1.as_str() },
                    "ExcitationTimes": excitation_times(p, out),
                    "ShotOrder": "s = interleaf * KzSegments + kz_segment, one RepetitionTimePreparation apart",
                    "VolumeDuration": r3.n_shots as f64 * tr0,
                    "Trajectory": {
                        "Kind": "Archimedean constant-density spiral-out, constant angular velocity inside CentreRegionMs, \
                                 constant linear velocity outside",
                        "ReadoutTimeMs": sp.readout_ms, "DwellTimeMs": sp.dwell_ms, "DwellTimeSource": sp.dwell_source.as_str(),
                        "SamplesPerInterleaf": sp.samples_per_interleaf, "KMax": sp.k_max, "Turns": sp.n_turns,
                        "RadialOversampling": { "Value": sp.radial_oversampling.0, "Source": sp.radial_oversampling.1.as_str(),
                                                "Meaning": "the turns are 1/RadialOversampling cycle/FOV apart across the interleaves" },
                        "CentreRegionMs": sp.tau_c_ms,
                        "SamplingBound": "k-space speed x dwell time <= 1 cycle/FOV on the continuous trajectory",
                    },
                    "TimeSegmentation": {
                        "Method": "least-squares interpolators on a tensor Chebyshev grid of the rate rectangle, \
                                   certified for every rate in it",
                        "Target": spiral_bound_target(),
                        "Class": fold(false), "Voxel": fold(true),
                    },
                    "Reconstruction": spiral_reconstruction_block(),
                    "SeedSalt": format!("{:#x}", mrsim_acq::kspace3d::SEED_SALT_3D),
                    "Noise": "per complex sample as Cartesian; the image noise after the least squares is measured, \
                              not asserted (about the Cartesian 3D value in the tests)",
                    "Approximations": approx,
                });
            } else {
            block["Readout"] = json!({
                "Type": rs.kind.0.as_str(), "TypeSource": rs.kind.1.as_str(),
                "NumberShots": r3.n_shots, "KySegments": rs.ky_segments.0, "KzSegments": rs.kz_segments.0,
                "EchoTrainLength": r3.etl, "LinesPerEcho": r3.epi,
                "EchoSpacingMs": r3.esp_ms,
                "EchoSpacingSource": if rs.echo_spacing_ms.is_some() { "overlay readout.echo_spacing" }
                                     else { "EchoTime, the k-space-centre time: CentreEcho x ESP + CentreLineTimeMs" },
                "LineSpacingMs": r3.t_line_ms, "LineSpacingSource": r3.t_line_source,
                "EffectiveEchoSpacing": r3.effective_spacing_s.unwrap_or(r3.t_line_ms / 1000.0 / rs.ky_segments.0 as f64),
                "KzOrder": match rs.kz_order.0 { mrsim_acq::readout::KzOrder::Centric => "centric", _ => "linear" },
                "CentreEcho": r3.e_c, "CentreLineTimeMs": r3.t_kyc_ms,
                "RefocusingFlipAngle": { "Value": rs.refocusing_flip_deg.0, "Source": rs.refocusing_flip_deg.1.as_str() },
                "FlipAngleInterpretation": "the sidecar's FlipAngle is the echo train's refocusing angle; the excitation is 90 degrees",
                "RefocusingTimeMs": { "Value": rs.refocusing_time_ms.0, "Source": rs.refocusing_time_ms.1.as_str() },
                "ExcitationTimes": excitation_times(p, out),
                "ShotOrder": "s = ky_segment * KzSegments + kz_segment, one RepetitionTimePreparation apart",
                "VolumeDuration": r3.n_shots as f64 * tr0,
                "SeedSalt": format!("{:#x}", mrsim_acq::kspace3d::SEED_SALT_3D),
                "Noise": "the image noise SD is 1/sqrt(nz) times a 2D acquisition's at the same noise_variance \
                          (linear Cartesian reconstruction; measured, not asserted, under GRAPPA)",
                "Approximations": approx,
            });
            }
            if let Some(amps) = &out.echo_amplitudes {
                // the echo reading each partition
                let echo_of: Option<Vec<usize>> = if r3.spiral.is_some() {
                    mrsim_acq::readout::spiral_lines(&r3.train, &r3.readout, ny, ny, nz).ok().map(|t| t.echo)
                } else {
                    mrsim_acq::readout::grase_lines(&r3.train, &r3.readout, ny, nz).ok()
                        .map(|t| (0..nz).map(|q| t.line(q, 0).echo).collect())
                };
                let mut labels = Map::new();
                let mut kz = Map::new();
                for (name, a) in amps {
                    labels.insert(name.clone(), json!(a));
                    if let Some(e) = &echo_of {
                        kz.insert(name.clone(), json!(e.iter().map(|&x| a[x - 1]).collect::<Vec<f64>>()));
                    }
                }
                block["EchoAmplitudes"] = json!({
                    "PerEcho": labels,
                    "KzModulation": kz,
                    "Note": if out.mode == T2Mode::Voxel { "the first voxel of each label (voxel mode: indicative)" }
                            else { "per label (class mode)" },
                });
            }
            // series' rule: the extravascular label has its own group in 3D under physio and exchange
            if p.physio.is_some() && p.exchange_time.is_some() {
                block["CompartmentOrder"] = json!(format!("{}, then the extravascular label per label (its own group in 3D \
                    under physiological noise and exchange)", block["CompartmentOrder"].as_str().unwrap_or("")));
            }
        }
        // P5 part A: written only under gradient echo, so other outputs keep their bytes
        if let Some(g) = &p.ge {
            block["GradientEcho"] = json!({
                "ExcitationFlipAngle": { "Value": g.flip_deg, "Source": g.flip.as_str() },
                "EchoFormation": "gradient echo: T2' decays from the RF (T2* = 1/(1/T2 + 1/T2')), and the static \
                                  2 pi fmap TE joins the object phase",
                "TissueModel": if p.compat.is_some() {
                    "simasl's steady state, keeping transverse coherence through exp(-TR/T2) (compat)"
                } else {
                    "spoiled steady state: sin(a) M0 (1 - E1) / (1 - cos(a) E1), or the suppression timeline's \
                     fixed point, or the state carried row to row when the rows' preparations differ"
                },
                "SteadyState": out.ge_rule,
                "BloodModel": "sin(FlipAngle) * delta_m",
                "M0": "the same excitation and readout without labeling",
            });
            block["M0ScanContrast"] = json!("ge");
        }
        if let (Some(_), Some(f)) = (&p.compat, &out.compat) {
            let mut pinned: Map<String, Value> = COMPAT_PINNED.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
            pinned.insert("window".to_string(), json!("none"));
            pinned.insert("readout_relaxation".to_string(), json!(false));
            block["M0ScanContrast"] = json!(p.contrast.as_str());
            block["Resolved"]["T2Blood"]["Used"] = json!(false);
            block["Compat"] = json!({
                "Asldro": true,
                "Reference": "simasl (ASLDRO v2.2.0); mrsim-acq/docs/specs/2026-09-24-p2-asldro-compat-design.md",
                "Pinned": pinned,
                "DesiredSnr": f.desired_snr,
                "NoiseVariance": f.noise_variance,
                "M0ReferenceMean": f.m0_reference_mean,
                "M0ReferenceVoxels": f.m0_reference_voxels,
                "M0Reference": "mean |M0| over the nonzero voxels of the box-averaged M0 ground truth on the \
                                acquisition grid; simasl's is its spline-resampled m0, so equal SNRs are not \
                                equal noise unless the grids coincide",
                "GridOrigin": p.grid_origin.as_str(),
                "RelaxationAtEcho": if p.contrast == crate::mrsignal::Contrast::GradientEcho {
                    "exp(-EchoTime/T2*) per phantom voxel, tissue and blood alike, T2* = 0 giving 1; \
                     no relaxation in the readout"
                } else {
                    "exp(-EchoTime/T2) per phantom voxel, tissue and blood alike, T2 = 0 giving 1; \
                     no relaxation in the readout"
                },
                "T2BloodUnused": true,
                "SliceOffsetsAllZero": true,
                "M0ScanRows": "the series' signal equation without label, as simasl's",
            });
        }
        block
    }

    /// The P4 sidecar blocks (addendum, "Outputs"), each only when its part is on.
    fn p4_blocks(block: &mut Value, p: &Protocol, out: &SeriesOutput) {
        if let Some(te) = p.exchange_time {
            block["Exchange"] = json!({
                "ExchangeTime": te,
                "Model": "single pass, irreversible, applied to the GKM kernel per parcel",
                "Placement": "intravascular label in the blood compartments, exchanged label in the tissue ones",
            });
        }
        if let Some(m) = &p.macrovascular {
            let q = |s: &QuantitySource, map: &str| match s {
                QuantitySource::Map => json!({ "Map": map }),
                QuantitySource::Table(t) => json!(t),
            };
            block["Macrovascular"] = json!({
                "ArterialBloodVolume": q(&m.abv, "abv.nii.gz"),
                "ArterialTransitTime": q(&m.aatt, "aatt.nii.gz"),
                "T2Arterial": resolved(m.t2_arterial),
                "Model": "plug flow, no dispersion (Chappell et al. 2010, the macrovascular term)",
            });
            block["CompartmentOrder"] = json!("tissue per label, then labeled blood per label, then arterial blood per label");
        }
        if let Some(c) = &p.crushing {
            let mut v = json!({
                "Venc": c.venc,
                "Model": "isotropic laminar: c = Si(pi r) / (pi r), r = v_max / VENC",
                "VencConvention": "phase pi at VENC (assumed; BIDS defines VascularCrushingVENC as a strength in cm/s)",
                "Unmodeled": ["crusher eddy currents", "crusher bulk-motion phase", "capillary flow"],
                "EddyDrive": Value::Null,
                "PrepPhase": Value::Null,
            });
            match (&c.arterial_velocity, &out.crush_survival) {
                (Some(vel), Some(surv)) => {
                    v["ArterialVelocity"] = json!(vel);
                    v["Survival"] = json!(surv);
                    v["SurvivalOrder"] = json!("per row, per label in Labels order");
                }
                _ => {
                    v["NoArterialCompartment"] = json!(true);
                    v["Note"] = json!("no arterial compartment: the crushers act on nothing modeled, and the data are \
                                       the uncrushed simulation");
                }
            }
            block["VascularCrushing"] = v;
        }
        if let Some(s) = &p.suppression {
            if let SuppressionModel::BolusPosition(region) = s.model {
                let (name, entry) = match region {
                    Region::Global => ("global", Value::Null),
                    Region::Slab(d) => ("slab", json!(d)),
                    Region::Arrival => ("slab", json!("arrival")),
                };
                if let Some(o) = block.as_object_mut() {
                    // no single factor per row under this model: the ground truth carries it
                    o.remove("BackgroundSuppressionLabelFactor");
                }
                block["BackgroundSuppressionModel"] = json!("bolus-position");
                block["BackgroundSuppression"] = json!({
                    "Model": "bolus-position",
                    "ModelNote": "a pulse acts on a parcel of label once it is inside the pulse's region; the \
                                  bolus is a sum of sub-boluses weighted by their factors (P4 addendum, part D)",
                    "PulseRegion": name,
                    "SlabEntryTime": entry,
                    "InversionEfficiency": resolved(s.epsilon),
                    "Presaturation": { "Value": s.presaturation.0, "Source": s.presaturation.1.as_str() },
                    "FirstPldPulseTimesAppliedToAll": s.first_pld_applied_to_all,
                    "PulseTimesPerRow": s.per_row,
                    "TissueModel": "signed longitudinal timeline per acquired slice; m0scan rows unsuppressed",
                    "EffectiveFactor": "desc-deltamSuppressed_gt over desc-deltam_gt, per voxel and row",
                });
            }
        }
        if let Some(ph) = &p.physio {
            block["Physio"] = json!({
                "TissueAmplitudes": { "Cardiac": ph.tissue[0], "Respiratory": ph.tissue[1], "Drift": ph.tissue[2] },
                "LabelAmplitudes": { "Cardiac": ph.label[0], "Respiratory": ph.label[1], "Drift": ph.label[2] },
                "CardiacFrequency": ph.cardiac_frequency, "CardiacCv": ph.cardiac_cv,
                "RespiratoryFrequency": ph.respiratory_frequency, "RespiratoryCv": ph.respiratory_cv,
                "DriftTime": ph.drift_time, "DriftGridStep": crate::physio::DRIFT_STEP,
                "Seed": p.seed, "SeedSalt": format!("{:#x}", crate::physio::PHYSIO_SEED_SALT),
                "Model": "global factors 1 + a_c sin(phi_c) + a_r sin(phi_r) + a_d x: the tissue's at each slice's \
                          readout, the label's averaged over the labeling window ((P)CASL) or at labeling (PASL)",
                "GroundTruth": "desc-physio_gt.tsv",
            });
        }
    }

    /// Write the whole dataset under `root`.
    pub fn write_dataset(root: &Path, names: &Names, p: &Protocol, out: &SeriesOutput) -> Result<(), String> {
        let perf = root.join(names.perf_dir());
        let gt_dir = perf.join("ground-truth");
        std::fs::create_dir_all(&gt_dir).map_err(|e| format!("{}: {e}", gt_dir.display()))?;
        let stem = names.stem();
        let prefix = perf.join(&stem);
        let prefix_s = prefix.to_string_lossy().to_string();

        // Dataset-level files.
        write_json(&root.join("dataset_description.json"), &json!({
            "Name": "aslscan simulation",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "Authors": ["aslscan"],
            "GeneratedBy": [{ "Name": "aslscan", "Version": env!("CARGO_PKG_VERSION"),
                              "Description": "Simulated ASL on the mrsim-acq acquisition stage" }],
        }))?;
        std::fs::write(root.join("README"), format!(
            "Simulated ASL dataset written by aslscan {}.\n\nThe protocol is the input BIDS ASL sidecar; \
             every value the simulator resolved beyond it is recorded under \"AslscanSimulation\" in the \
             *_asl.json sidecars. Ground-truth maps on the acquisition grid are under perf/ground-truth/ \
             (listed in .bidsignore).\n", env!("CARGO_PKG_VERSION")))
            .map_err(|e| e.to_string())?;
        // `**/ground-truth` without a trailing slash is the form bids-validator 3.0.2 honours for
        // the directory; the `/`-suffixed and `/**` forms left NOT_INCLUDED errors behind.
        std::fs::write(root.join(".bidsignore"), "**/ground-truth\n**/ground-truth/**\n")
            .map_err(|e| e.to_string())?;

        // The complex pair through the shared writer, then the sidecars rewritten in ASL terms.
        // The NIfTI time step is one number; with per-row repetition times (an included M0 row,
        // say) it is the first ASL row's, and the sidecar's RepetitionTimePreparation array is
        // the authority.
        let nifti_tr = p.rows.iter().find(|r| r.kind != crate::rows::RowKind::M0scan).or(p.rows.first()).map(|r| r.tr);
        // a segmented 3D volume takes NumberShots repetitions: the time step is the volume's (P5
        // part D); the sidecar's RepetitionTimePreparation stays the per-shot value given
        let nifti_tr = match &out.readout {
            Some(r3) => nifti_tr.map(|tr| r3.n_shots as f64 * tr),
            None => nifti_tr,
        };
        // P6 part C: one series per echo (`echo-N`), each from its own input sidecar; with one echo the
        // names carry no echo entity and this runs once, as before
        let n_echo = p.echo_times_s.len();
        if out.more_echoes.len() + 1 != n_echo {
            return Err(format!("{n_echo} echo times but {} echo series simulated", out.more_echoes.len() + 1));
        }
        for e in 0..n_echo {
            let echo_tag = if n_echo > 1 { format!("_echo-{}", e + 1) } else { String::new() };
            let prefix_e = format!("{prefix_s}{echo_tag}");
            let input_e: &Value = match &p.multi_te {
                Some(m) => &m.sidecars[e],
                None => &p.input_sidecar,
            };
            let echo_time_s = p.echo_times_s[e];
            let (mag_e, phase_e, m0_e) = if e == 0 {
                (&out.mag[..], &out.phase[..], out.m0.as_ref())
            } else {
                let x = &out.more_echoes[e - 1];
                (&x.mag[..], &x.phase[..], x.m0.as_ref())
            };
            let info = SidecarInfo {
                manufacturer: "aslscan".to_string(),
                phase_encoding_direction: p.phase_encoding_direction.clone(),
                total_readout_time: p.total_readout_time_s,
                echo_time: echo_time_s,
                partial_fourier: out.acquisition.partial_fourier,
                accel: out.acquisition.accel,
                mb: p.mb,
                repetition_time_s: nifti_tr,
                b0_field_source: None,
            };
            write_complex_4d(&prefix_e, "asl", out.acq_grid.dims, out.n_volumes, mag_e, phase_e, &out.acq_grid, &info)
                .map_err(|e| e.to_string())?;
            // The shared writer's part sidecars are DWI-flavoured stubs. Each part gets the complete
            // ASL sidecar (the phase part with its Units). There is deliberately NO inheritance-level
            // `_asl.json`: with `part-` entities there is no `_asl.nii.gz`, and bids-validator 3
            // flags such a file as SIDECAR_WITHOUT_DATAFILE; it also checks required keys per part
            // file without merging a less specific sidecar in, so each part must be complete.
            let mut side: Map<String, Value> = input_e.as_object().cloned().unwrap_or_default();
            // Standard keys describe what was SIMULATED, not what the input said: an overlay may
            // have overridden the sidecar's LabelingEfficiency, and PartialFourier or the
            // acceleration factor come from the overlay/protocol, not the input. The originals are
            // kept under AslscanSimulation.InputValuesReplaced so nothing is lost.
            let mut effective: Vec<(&str, Value)> = vec![
                ("LabelingEfficiency", json!(p.alpha.0)),
                ("PartialFourier", json!(out.acquisition.partial_fourier)),
                ("ParallelReductionFactorInPlane", json!(out.acquisition.accel)),
                ("MultibandAccelerationFactor", json!(p.mb)),
                ("TotalAcquiredPairs", json!(p.hadamard.as_ref().map_or(p.total_acquired_pairs(), |h| h.cycles.len()))),
            ];
            if let Some(ir) = &p.ir {
                // Standard fields; a simasl-legal negative excitation angle is written as its
                // positive equivalent (BIDS: 0..360), the signed value staying in the block above.
                effective.push(("InversionTime", json!(ir.params.inversion_time)));
                effective.push(("FlipAngle", json!(ir.params.excitation_flip_deg.rem_euclid(360.0))));
            }
            if let Some(g) = &p.ge {
                // P6 part B: a Look-Locker FlipAngle array is every volume's own excitation
                let fa = match &p.look_locker {
                    Some(l) if l.flip_array => json!(l.flip_deg),
                    _ => json!(g.flip_deg.rem_euclid(360.0)),
                };
                effective.push(("FlipAngle", fa));
            }
            // P5 part B: a 3D readout's standard keys as resolved (the effective spacing BIDS defines,
            // TotalReadoutTime = it x (ny - 1), the direction, the shots, the refocusing angle)
            if let (Some(r3), Some(rs)) = (&out.readout, &p.readout) {
                if let Some(sp) = &r3.spiral {
                    // a spiral has no phase-encode readout (its EES, TRT and PED were refused)
                    effective.push(("DwellTime", json!(sp.dwell_ms / 1000.0)));
                } else {
                    let ny = out.acq_grid.dims[1];
                    let ees = r3.effective_spacing_s.unwrap_or(r3.t_line_ms / 1000.0 / rs.ky_segments.0 as f64);
                    effective.push(("EffectiveEchoSpacing", json!(ees)));
                    effective.push(("TotalReadoutTime", json!(ees * (ny as f64 - 1.0))));
                    effective.push(("PhaseEncodingDirection", json!(p.phase_encoding_direction)));
                }
                if let Some(t) = overridden_sequence_type(p) {
                    effective.push(("PulseSequenceType", json!(t)));
                }
                effective.push(("NumberShots", json!(r3.n_shots)));
                effective.push(("FlipAngle", json!(rs.refocusing_flip_deg.0)));
            }
            if let Some(s) = &p.suppression {
                // BIDS carries the first PLD's pulse times; an overlay override must be what is
                // published, with the input kept under InputValuesReplaced.
                effective.push(("BackgroundSuppressionNumberPulses", json!(s.first_pld_pulses.len())));
                effective.push(("BackgroundSuppressionPulseTime", json!(s.first_pld_pulses)));
            }
            let mut replaced = Map::new();
            for (k, v) in effective {
                if let Some(old) = side.get(k) {
                    if !same_number(old, &v) {
                        replaced.insert(k.to_string(), old.clone());
                    }
                }
                side.insert(k.to_string(), v);
            }
            let mut sim = simulation_block(p, out);
            if e > 0 {
                // the acquisition record of this echo (the one Acquisition is echo 1's)
                sim["Acquisition"]["TEchoMs"] = json!(echo_time_s * 1000.0);
            }
            if let Some(hb) = hadamard_block(p, out, e) {
                sim["Hadamard"] = hb;
            }
            if let Some(lb) = look_locker_block(p, out) {
                sim["LookLocker"] = lb;
            }
            sim["InputValuesReplaced"] = Value::Object(replaced);
            if let Some(me) = multi_echo_block(p, out, e) {
                sim["MultiEcho"] = me;
            }
            side.insert("AslscanSimulation".to_string(), sim);
            write_json(&PathBuf::from(format!("{prefix_e}_part-mag_asl.json")), &Value::Object(side.clone()))?;
            side.insert("Units".to_string(), json!("rad"));
            write_json(&PathBuf::from(format!("{prefix_e}_part-phase_asl.json")), &Value::Object(side))?;
            // one aslcontext for every echo (BIDS inheritance: it carries no echo entity)
            if e == 0 {
                std::fs::write(format!("{prefix_s}_aslcontext.tsv"), aslcontext_tsv(&p.rows)).map_err(|e| e.to_string())?;
            }

            // The separate M0 scan.
            if p.m0_type == M0Type::Separate {
                let (mag, _phase) = m0_e.ok_or("M0Type Separate but no M0 volume was simulated")?;
                write_3d(&PathBuf::from(format!("{prefix_e}_m0scan.nii.gz")), out.acq_grid.dims, mag, &out.acq_grid)
                    .map_err(|e| e.to_string())?;
                let mut m0side = Map::new();
                // The readout and the hardware are the ASL series'; BIDS recommends the hardware keys
                // on every sidecar, so they are carried over when the input has them.
                for k in ["Manufacturer", "ManufacturersModelName", "DeviceSerialNumber", "StationName",
                          "SoftwareVersions", "MagneticFieldStrength", "ReceiveCoilName", "ReceiveCoilActiveElements",
                          "GradientSetType", "MRTransmitCoilSequence", "MatrixCoilMode", "CoilCombinationMethod",
                          "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
                          "MRAcquisitionType", "PhaseEncodingDirection", "TotalReadoutTime", "EchoTime",
                          "SliceTiming", "SliceEncodingDirection", "AcquisitionVoxelSize", "FlipAngle"] {
                    if let Some(v) = input_e.get(k) {
                        m0side.insert(k.to_string(), v.clone());
                    }
                }
                // The M0 scan shares the ASL readout, so its effective readout values are the same.
                m0side.insert("PartialFourier".to_string(), json!(out.acquisition.partial_fourier));
                m0side.insert("ParallelReductionFactorInPlane".to_string(), json!(out.acquisition.accel));
                m0side.insert("MultibandAccelerationFactor".to_string(), json!(p.mb));
                m0side.insert("RepetitionTimePreparation".to_string(), json!(p.m0_repetition_time_s));
                m0side.insert("IntendedFor".to_string(), json!([
                    format!("bids::{}", names.rel(&format!("{echo_tag}_part-mag_asl.nii.gz"))),
                    format!("bids::{}", names.rel(&format!("{echo_tag}_part-phase_asl.nii.gz"))),
                ]));
                // The M0 scan is simulated with the 90-degree spin-echo equation whatever the ASL
                // series' excitation angle, except under gradient echo, whose M0 is the same
                // excitation and readout (P5 part A); its FlipAngle says which, the input kept as
                // replaced.
                let (m0_flip, m0_contrast, m0_note) = match (&p.ge, &p.readout) {
                    // P6 part B: a Look-Locker series' M0 at its own excitation, not a Look-Locker file
                    (Some(_), _) if p.look_locker.as_ref().is_some_and(|l| l.m0_flip_deg.is_some()) => (
                        p.look_locker.as_ref().and_then(|l| l.m0_flip_deg).map_or(90.0, |f| f.0), "ge",
                        "a gradient-echo readout at its own repetition time and excitation (overlay m0.flip_angle, or the \
                     series' scalar FlipAngle): one excitation, not a Look-Locker series; no suppression, no motion"),
                    (Some(g), _) => (g.flip_deg.rem_euclid(360.0), "ge",
                                     "the series' gradient-echo readout at its own repetition time: no suppression, no motion"),
                    // P5 part B: the same echo train, its FlipAngle the refocusing angle
                    (None, Some(rs)) => (rs.refocusing_flip_deg.0, "se",
                                         "the series' 3D echo train at its own repetition time, excited at its start: no \
                                          labeling, no suppression, no motion"),
                    (None, None) => (90.0, "se", "a plain spin-echo readout at its own repetition time: no suppression, no inversion, no motion"),
                };
                if let (Some(r3), Some(rs)) = (&out.readout, &p.readout) {
                    if let Some(sp) = &r3.spiral {
                        m0side.insert("DwellTime".to_string(), json!(sp.dwell_ms / 1000.0));
                    } else {
                        let ny = out.acq_grid.dims[1];
                        let ees = r3.effective_spacing_s.unwrap_or(r3.t_line_ms / 1000.0 / rs.ky_segments.0 as f64);
                        m0side.insert("EffectiveEchoSpacing".to_string(), json!(ees));
                        m0side.insert("TotalReadoutTime".to_string(), json!(ees * (ny as f64 - 1.0)));
                        m0side.insert("PhaseEncodingDirection".to_string(), json!(p.phase_encoding_direction));
                    }
                    m0side.insert("NumberShots".to_string(), json!(r3.n_shots));
                    if let Some(t) = overridden_sequence_type(p) {
                        m0side.insert("PulseSequenceType".to_string(), json!(t));
                    }
                }
                let mut m0_replaced = Map::new();
                if let Some(old) = m0side.get("FlipAngle") {
                    if !same_number(old, &json!(m0_flip)) {
                        m0_replaced.insert("FlipAngle".to_string(), old.clone());
                    }
                }
                // the shared readout's values the M0 sidecar states in place of the input's
                for k in ["DwellTime", "PulseSequenceType"] {
                    if let (Some(old), Some(new)) = (input_e.get(k), m0side.get(k)) {
                        if !same_number(old, new) {
                            m0_replaced.insert(k.to_string(), old.clone());
                        }
                    }
                }
                m0side.insert("FlipAngle".to_string(), json!(m0_flip));
                let mut m0sim = json!({
                    "Seed": out.seeds.1, "Magnitude": true, "Contrast": m0_contrast,
                    "Note": m0_note,
                    "InputValuesReplaced": m0_replaced,
                });
                if let (Some(m), Some(seed)) = (&p.multi_te, out.seeds.1) {
                    m0sim["MultiEcho"] = json!({
                        "Echo": e + 1, "EchoTimes": p.echo_times_s, "ExcitationSeed": seed,
                        "ReceiverSeed": seed ^ mrsim_acq::kspace::echo_salt(e), "Echoes": m.sidecars.len(),
                    });
                }
                if p.physio.is_some() {
                    m0sim["Physio"] = json!("not applied: the separate M0 scan is not a row of the series' clock");
                }
                if let Some(r3) = &out.readout {
                    // the series' readout block without its excitation times (the M0 is excited at the
                    // start of its own repetition)
                    let mut ro = simulation_block(p, out)["Readout"].clone();
                    if let Some(o) = ro.as_object_mut() {
                        o.remove("ExcitationTimes");
                        o.insert("VolumeDuration".to_string(), json!(r3.n_shots as f64 * p.m0_repetition_time_s.unwrap_or(0.0)));
                    }
                    m0sim["Readout"] = ro;
                }
                m0side.insert("AslscanSimulation".to_string(), m0sim);
                write_json(&PathBuf::from(format!("{prefix_e}_m0scan.json")), &Value::Object(m0side))?;
            }
        }

        // Ground truth.
        let gt = &out.ground_truth;
        let gt_prefix = gt_dir.join(&stem).to_string_lossy().to_string();
        let wgt = |desc: &str, data: &[f32], units: &str, how: &str| -> Result<(), String> {
            write_3d(&PathBuf::from(format!("{gt_prefix}{}", ground_truth_tail(desc))), out.acq_grid.dims, data, &out.acq_grid)
                .map_err(|e| e.to_string())?;
            write_json(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.json")),
                       &json!({ "Units": units, "Resampling": how }))
        };
        let mean = "volume-weighted mean over the phantom voxels each acquisition voxel overlaps";
        wgt("perfusion", &gt.perfusion, "ml/100g/min", mean)?;
        wgt("att", &gt.att, "s", "volume-weighted mean over perfused (perfusion > 0) phantom voxels; 0 where none")?;
        wgt("T1map", &gt.t1, "s", mean)?;
        wgt("T2map", &gt.t2, "s", mean)?;
        wgt("M0map", &gt.m0, "arbitrary", mean)?;
        write_4d(&PathBuf::from(format!("{gt_prefix}_desc-deltam_gt.nii.gz")), out.acq_grid.dims, out.n_volumes, &gt.delta_m, &out.acq_grid)
            .map_err(|e| e.to_string())?;
        let moved = out.ground_truth.delta_m_static.is_some();
        write_json(&PathBuf::from(format!("{gt_prefix}_desc-deltam_gt.json")), &json!({
            "Units": "arbitrary (same as M0map)",
            "Description": if out.look_locker.as_ref().is_some_and(|l| !l.legacy_dispatch) {
                "+delta_m at each readout's own excitation per slice (e = t_n + slice offset), undepleted: the label \
                 delivered, before the readouts deplete it (desc-deltamRead_gt is what each readout read); \
                 box-averaged; zero for other rows"
            } else if out.hadamard.is_some() {
                "the ideal sub-bolus truth (P6 part A): each decoded volume's delta_m of its sub-bolus from the \
                 kinetics alone at the cycle's readout, without physiological or suppression factors, static (a \
                 decoded volume has no single pose), box-averaged; zero for m0scan rows. The raw volumes' truth is \
                 under sourcedata"
            } else if moved {
                "+delta_m at each label/deltam row's own timing, moved by the row's pose on the simulation grid \
                 and box-averaged (no shot events, no suppression factor); zero for other rows"
            } else {
                "+delta_m at each label/deltam row's own timing, box-averaged; zero for other rows"
            },
            "Resampling": mean,
            "Moved": moved,
        }))?;
        if let Some(gts) = &gt.delta_m_static {
            write_4d(&PathBuf::from(format!("{gt_prefix}_desc-deltamStatic_gt.nii.gz")), out.acq_grid.dims, out.n_volumes, gts, &out.acq_grid)
                .map_err(|e| e.to_string())?;
            write_json(&PathBuf::from(format!("{gt_prefix}_desc-deltamStatic_gt.json")), &json!({
                "Units": "arbitrary (same as M0map)",
                "Description": "+delta_m at each label/deltam row's own timing, unmoved, box-averaged; zero for other rows",
                "Resampling": mean,
            }))?;
        }
        // P4 ground truth (addendum, "Outputs"): each names the stages it includes.
        let frame = if moved { "moved by each row's pose like desc-deltam_gt" } else { "static" };
        for (desc, data, stages) in [
            ("deltamIntravascular", &gt.delta_m_iv, "kinetics and the exchange split (part A); no pulse or physiological factor"),
            ("deltamSuppressed", &gt.delta_m_suppressed, "kinetics and the bolus-position pulse factors (part D), both parts of the tissue label"),
            ("deltamArterial", &gt.delta_m_arterial, "the arterial term with parcel factor 1 (part B), before crushing"),
        ] {
            if let Some(d) = data {
                write_4d(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.nii.gz")), out.acq_grid.dims, out.n_volumes, d, &out.acq_grid)
                    .map_err(|e| e.to_string())?;
                write_json(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.json")), &json!({
                    "Units": "arbitrary (same as M0map)", "Stages": stages, "Frame": frame, "Resampling": mean,
                }))?;
            }
        }
        if let Some(a) = &gt.abv {
            wgt("aBV", a, "fraction", mean)?;
        }
        if let Some(a) = &gt.aatt {
            wgt("aATT", a, "s", "volume-weighted mean over phantom voxels with aBV > 0; 0 where none")?;
        }
        if out.hadamard.is_none() {
            write_volume_tables(&gt_prefix, p, out)?;
        }
        if let Some(l) = &out.look_locker {
            if let Some(d) = &l.delta_m_read {
                write_4d(&PathBuf::from(format!("{gt_prefix}_desc-deltamRead_gt.nii.gz")), out.acq_grid.dims, out.n_volumes, d, &out.acq_grid)
                    .map_err(|e| e.to_string())?;
                write_json(&PathBuf::from(format!("{gt_prefix}_desc-deltamRead_gt.json")), &json!({
                    "Units": "arbitrary (same as M0map)",
                    "Description": "what each Look-Locker readout read: sin(a_n) times the label depleted by the cycle's \
                                    earlier readouts (each leaving cos(a) of the label that had arrived in the slice), \
                                    per slice, static; zero for other rows",
                    "Resampling": mean,
                }))?;
            }
            // P7 part A: the read by part, each as desc-deltamRead_gt is (sin(a_n) times it)
            for (part, d, what) in [
                ("deltamReadIV", &l.read_iv, "the intravascular part of desc-deltamRead_gt: the label not yet \
                  exchanged, depleted as the whole is (the blood compartment)"),
                ("deltamReadEV", &l.read_ev, "the extravascular part of desc-deltamRead_gt: the exchanged label, \
                  depleted as the whole is (the tissue compartment); IV + EV = desc-deltamRead_gt"),
                ("arterialRead", &l.read_arterial, "the arterial compartment as each readout read it: sin(a_n) times \
                  the arterial delta-M at the slice's excitation with its crushing survival and parcel suppression \
                  factor, undepleted (fresh arterial blood)"),
            ] {
                if let Some(d) = d {
                    write_4d(&PathBuf::from(format!("{gt_prefix}_desc-{part}_gt.nii.gz")), out.acq_grid.dims, out.n_volumes, d, &out.acq_grid)
                        .map_err(|e| e.to_string())?;
                    write_json(&PathBuf::from(format!("{gt_prefix}_desc-{part}_gt.json")), &json!({
                        "Units": "arbitrary (same as M0map)",
                        "Description": format!("{what}; per slice, static; zero for other rows"),
                        "Resampling": mean,
                    }))?;
                }
            }
            if !l.lines.is_empty() {
                let mut tsv = String::from("cycle\treadout\tgroup\ttime\tflip_angle");
                for (_, name) in &out.labels {
                    tsv.push_str(&format!("\ttissue_mz_{name}"));
                }
                tsv.push('\n');
                for x in &l.lines {
                    tsv.push_str(&format!("{}\t{}\t{}\t{}\t{}", x.cycle, x.readout, x.group, x.time, x.flip_deg));
                    for m in &x.tissue_mz {
                        tsv.push_str(&format!("\t{m}"));
                    }
                    tsv.push('\n');
                }
                std::fs::write(format!("{gt_prefix}_desc-lookLocker_gt.tsv"), tsv).map_err(|e| e.to_string())?;
            }
        }
        // `phantom::load` already bounds labels to 0..=32767, so this cannot truncate; the
        // conversion is checked anyway rather than cast.
        let dseg: Vec<i16> = gt.dseg.iter().map(|l| i16::try_from(*l).map_err(|_| format!("dseg label {l} does not fit int16")))
            .collect::<Result<_, _>>()?;
        write_3d_i16(&PathBuf::from(format!("{gt_prefix}_desc-dseg_gt.nii.gz")), out.acq_grid.dims, &dseg, &out.acq_grid)
            .map_err(|e| e.to_string())?;
        write_json(&PathBuf::from(format!("{gt_prefix}_desc-dseg_gt.json")), &json!({
            "Units": "label indices", "Resampling": "majority vote by overlap, ties to the lower label",
            "LabelMap": out.labels.iter().map(|(l, n)| (l.to_string(), json!(n))).collect::<Map<String, Value>>(),
        }))?;
        if out.mode == T2Mode::Voxel {
            for (desc, data) in [("acqT2map", &gt.acq_t2_ms), ("acqT2primemap", &gt.acq_t2p_ms), ("acqT1map", &gt.acq_t1_ms)] {
                if let Some(d) = data {
                    write_3d(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.nii.gz")), out.sim_grid.dims, d, &out.sim_grid)
                        .map_err(|e| e.to_string())?;
                    write_json(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.json")), &json!({
                        "Units": "ms", "Grid": "simulation",
                        "Resampling": "M0-weighted mean of rates, inverted; INFINITY stored as inf",
                    }))?;
                }
            }
        }
        if let Some(h) = &out.hadamard {
            write_hadamard_sourcedata(root, names, p, out, h)?;
        }
        Ok(())
    }

    /// The physiology and motion tables, one line per volume (and slice or shot): the series'
    /// own, or under Hadamard the raw volumes', which sourcedata carries.
    fn write_volume_tables(gt_prefix: &str, p: &Protocol, out: &SeriesOutput) -> Result<(), String> {
        if let Some(lines) = &out.physio {
            // 2D: per (volume, slice), unchanged; 3D: per (volume, shot), at each shot's excitation
            // (P5 part D), a branch of its own so the 2D file keeps its bytes
            let mut tsv = if out.readout.is_some() {
                String::from(
                    "volume\tshot\ttime\tcardiac_phase\trespiratory_phase\tdrift\ttissue_factor\tlabel_window_start\t\
                     label_window_end\tlabel_mean_sin_cardiac\tlabel_mean_sin_respiratory\tlabel_mean_drift\tlabel_factor\n")
            } else {
                String::from(
                    "volume\tslice\ttime\tcardiac_phase\trespiratory_phase\tdrift\ttissue_factor\tlabel_window_start\t\
                     label_window_end\tlabel_mean_sin_cardiac\tlabel_mean_sin_respiratory\tlabel_mean_drift\tlabel_factor\n")
            };
            for l in lines {
                tsv.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", l.volume, l.slice, l.time, l.cardiac_phase,
                    l.respiratory_phase, l.drift, l.tissue_factor, l.label_window.0, l.label_window.1, l.label_means[0],
                    l.label_means[1], l.label_means[2], l.label_factor));
            }
            std::fs::write(format!("{gt_prefix}_desc-physio_gt.tsv"), tsv).map_err(|e| e.to_string())?;
        }
        if p.motion.is_some() {
            let mut tsv = String::from("volume\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n");
            for (v, q) in out.poses.iter().enumerate() {
                let r = q.rot_deg.map(f64::to_radians);
                tsv.push_str(&format!("{v}\t{}\t{}\t{}\t{}\t{}\t{}\n", q.trans_mm[0], q.trans_mm[1], q.trans_mm[2], r[0], r[1], r[2]));
            }
            std::fs::write(format!("{gt_prefix}_desc-motion_gt.tsv"), tsv).map_err(|e| e.to_string())?;
            let mut ev = String::from("volume\tshot\tslices\tattenuation\tjump_x\tjump_y\tjump_z\tjump_rx\tjump_ry\tjump_rz\n");
            for (e, d) in out.events.iter().zip(&out.dropped) {
                let slices = d.slices.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
                ev.push_str(&format!("{}\t{}\t{slices}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", e.volume, e.shot, d.attenuation,
                                     e.jump_mm[0], e.jump_mm[1], e.jump_mm[2],
                                     e.jump_deg[0].to_radians(), e.jump_deg[1].to_radians(), e.jump_deg[2].to_radians()));
            }
            std::fs::write(format!("{gt_prefix}_desc-motionEvents_gt.tsv"), ev).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The excitation times the 3D readout block records: the rows', or the raw volumes' under
    /// Hadamard (a decoded row's delay runs from its own sub-bolus).
    fn excitation_times(p: &Protocol, out: &SeriesOutput) -> Vec<f64> {
        match &out.hadamard {
            Some(h) => h.schedule.raw_rows.iter().map(|r| r.t).collect(),
            None => p.rows.iter().map(|r| r.t).collect(),
        }
    }

    /// The labeled sub-boli of an encoded raw volume (1-based), or none.
    fn labeled(order: usize, row: Option<usize>) -> Vec<usize> {
        match row {
            Some(i) => crate::hadamard::weights(&crate::hadamard::encoding(order)[i]).iter().enumerate()
                .filter(|(_, w)| **w == 1).map(|(j, _)| j + 1).collect(),
            None => Vec::new(),
        }
    }

    /// `AslscanSimulation.LookLocker` (P6 part B, "Outputs").
    fn look_locker_block(p: &Protocol, out: &SeriesOutput) -> Option<Value> {
        let (l, ls) = (p.look_locker.as_ref()?, out.look_locker.as_ref()?);
        let mut block = look_locker_block_p6(p, l, ls);
        // P7 part A: with a P4 part in force, the parts replace P6's list of refusals; without
        // one the block is P6's, byte for byte
        if !ls.p4_parts.is_empty() {
            let obj = block.as_object_mut().expect("an object");
            obj.remove("Refused");
            obj.insert("P4Parts".into(), json!(ls.p4_parts));
            obj.insert("P4PartsModel".into(), json!(
                "each part of the label is depleted as the whole is: exchange splits every arrival window's label \
                 into the part not yet exchanged (T1'' residue) and the rest; bolus-position suppression cuts the \
                 bolus into sub-boli, each with its parcel factor; crushing's VENC is per readout"));
            obj.insert("FreshArterial".into(), json!(
                "the arterial compartment holds the parcel passing through at each excitation, not depleted by earlier \
                 readouts (QUASAR's assumption; it holds when blood crosses a slice much faster than the readout \
                 spacing)"));
            obj.insert("DepletionBeforeArrival".into(), json!(
                "not modeled in 2D: label is depleted from its arrival in the voxel, not while it crosses other imaged \
                 slices"));
        }
        Some(block)
    }

    /// P6's `AslscanSimulation.LookLocker`.
    fn look_locker_block_p6(p: &Protocol, l: &crate::protocol::LookLockerSpec, ls: &LookLockerSeries) -> Value {
        let cycles: Vec<Value> = l.cycles.iter().map(|c| json!({
            "FirstRow": c.rows[0],
            "Readouts": c.rows.len(),
            "M0scan": c.m0scan,
            "ExcitationTimes": if c.m0scan { vec![0.0] } else { c.rows.iter().map(|&r| p.rows[r].t).collect::<Vec<f64>>() },
            "FlipAngles": c.rows.iter().map(|&r| l.flip_deg[r]).collect::<Vec<f64>>(),
        })).collect();
        json!({
            "Cycles": cycles,
            "ReadoutsPerCycle": l.readouts_per_cycle,
            "LegacyDispatch": ls.legacy_dispatch,
            "LegacyDispatchNote": "one readout per cycle at one flip is P5's gradient-echo series itself, bit for bit",
            "TissueModel": "per slice, the cycle's events on Mz: the suppression pulses before the first readout, then \
                            each readout (its signal sin(a) Mz just before it, leaving cos(a) Mz), T1 recovery between \
                            them and to TR; the steady state of the repeated cycle is its affine fixed point, and cycles \
                            that differ carry the state forward; an m0scan cycle is one readout at the start of its own \
                            repetition",
            "BloodModel": "each readout depletes the difference magnetization of the label that has arrived in its slice \
                           (label in transit is outside the imaged slices); the read is sin(a_n) sum_k [prod cos(a_m)] \
                           delta_m_arrival over the arrival windows between the slice's excitations, with the GKM's T1' \
                           after arrival",
            "M0FlipAngle": l.m0_flip_deg.map(|(v, src)| json!({ "Value": v, "Source": src.as_str() })),
            "Refused": ["exchange (P4 part A)", "the arterial compartment (P4 part B)", "crushing (P4 part C)",
                        "bolus-position suppression (P4 part D)"],
            "PoseIndex": "motion poses are indexed by volume (each readout its own), not by time",
            "Pixdim4": "the NIfTI time step is the first non-m0scan row's RepetitionTimePreparation, a storage \
                        convention; the readout schedule is Cycles",
            "GroundTruth": if ls.legacy_dispatch {
                "desc-deltam_gt: the delta_m at each slice's excitation, which one readout per cycle reads undepleted; the \
                 legacy dispatch writes no desc-deltamRead_gt or desc-lookLocker_gt.tsv"
            } else {
                "desc-deltam_gt: the undepleted delta_m at each slice's excitation; desc-deltamRead_gt: what each readout \
                 read; desc-lookLocker_gt.tsv: one line per readout and excitation group with the mean tissue Mz before \
                 the pulse per phantom label"
            },
        })
    }

    /// `AslscanSimulation.Hadamard` (P6 part A, "Outputs").
    fn hadamard_block(p: &Protocol, out: &SeriesOutput, e: usize) -> Option<Value> {
        let (h, hs) = (p.hadamard.as_ref()?, out.hadamard.as_ref()?);
        let sched = &hs.schedule;
        // echo e's leakage, acquired at its own TE
        let echo_leakage = if e == 0 { hs.leakage.clone() } else { hs.leakage_more_echoes.get(e - 1).cloned() };
        let leakage = match &echo_leakage {
            Some(l) => json!(l.iter().map(|x| {
                let mut v = json!({
                    "Cycle": x.cycle + 1,
                    "ReferenceNorm": x.reference_norm,
                    "Absolute": x.per_subbolus.iter().map(|q| q.0).collect::<Vec<f64>>(),
                    "Normalized": x.per_subbolus.iter().map(|q| q.1).collect::<Vec<f64>>(),
                });
                // P7 part B: each readout decoded on its own
                if h.readouts > 1 {
                    v["Readout"] = json!(x.readout + 1);
                }
                v
            }).collect::<Vec<_>>()),
            None => json!("not computed (hadamard.report_leakage = false)"),
        };
        let f = hs.flags;
        let mut block = json!({
            "Order": h.order,
            "Matrix": crate::hadamard::encoding(h.order),
            "MatrixNote": "Sylvester, the all-ones first column removed; row i is raw volume i of a cycle, column j \
                           sub-bolus j + 1 (labeled first); -1 labels the sub-bolus, +1 leaves it as control",
            "SubBoli": h.spans.iter().map(|(a, b)| [*a, *b]).collect::<Vec<_>>(),
            "LabelingDuration": h.tau_tot,
            "PostLabelingDelay": h.pld,
            "ReportLeakage": resolved_bool(h.report_leakage),
            "Cycles": h.cycles.iter().map(|c| c.rows.clone()).collect::<Vec<_>>(),
            "Outputs": sched.outputs.iter().map(|o| match *o {
                crate::schedule::Output::Raw(r) => json!({ "RawVolume": r }),
                crate::schedule::Output::Decoded { cycle, subbolus, readout } if h.readouts > 1 => {
                    json!({ "Cycle": cycle + 1, "SubBolus": subbolus + 1, "Readout": readout + 1 })
                }
                crate::schedule::Output::Decoded { cycle, subbolus, .. } => json!({ "Cycle": cycle + 1, "SubBolus": subbolus + 1 }),
            }).collect::<Vec<_>>(),
            "RawVolumes": sched.raws.iter().map(|r| json!({
                "Cycle": r.cycle.map(|c| c + 1), "EncodingRow": r.encoding_row,
                "LabeledSubBoli": labeled(h.order, r.encoding_row),
                "FirstPreparation": r.prep, "Preparations": r.n_preps,
            })).collect::<Vec<_>>(),
            "Counts": { "Preparations": sched.preps.len(), "RawVolumes": sched.raws.len(),
                        "Decoded": sched.outputs.iter().filter(|o| matches!(o, crate::schedule::Output::Decoded { .. })).count() },
            "DecodingRule": "D_j = (2/H) sum_i h_ij S_i over each cycle's raw volumes S_i, on the complex images (from \
                             the acquisition's float32 magnitude and phase) in float64; the label subtracts, so D_j has \
                             the sign of a deltam volume; m0scan rows are raw volumes passed through",
            "NoiseScale": 2.0 / (h.order as f64).sqrt(),
            "NoiseScaleNote": "the decoded noise SD is the raw SD times 2/sqrt(H) (measured, not imposed)",
            "TissueLeakage": leakage,
            "TissueLeakageDefinition": "L_j: the decoded tissue-only raw volumes (the tissue kept apart before any \
                                        label is added, so the extravascular label is not counted), acquired with noise, \
                                        spikes and GRAPPA off; ||L_j||_2 over the brain mask, absolute and over the norm of \
                                        the cycle's unsuppressed tissue steady state acquired in the same call",
            "NonExact": { "Grappa": f.grappa, "Spikes": f.spikes, "Motion": f.motion, "ShotFactors": f.shot_factors,
                          "Transients": f.transients, "Physiology": f.physiology },
            "Readout": match &out.readout {
                Some(r3) if r3.spiral.is_some() => "3D spiral",
                Some(_) => "3D GRASE",
                None => "2D EPI",
            },
            "TotalAcquiredPairsConvention": "the number of encoding cycles: each cycle measures every sub-bolus once, \
                                             the role a control-label pair plays for one PLD; a Hadamard acquisition has \
                                             no control-label pairs",
            "GroundTruth": "this dataset: the ideal sub-bolus truth per decoded volume (static); sourcedata: the raw \
                            volumes' encoded truth and desc-preparations_gt.tsv (the factors applied per preparation)",
            "PerVolumeRecords": "the per-volume entries of this block's siblings (background-suppression label factors, \
                                 crushing survival, motion, physiology) index the raw volumes listed in RawVolumes",
        });
        // P7 part B: each encoded preparation read by M Look-Locker readouts
        if h.readouts > 1 {
            block["Readouts"] = json!(h.readouts);
            block["PostLabelingDelays"] = json!(h.plds);
            block["ReadoutDecoding"] = json!(
                "each readout index decoded on its own: output (j, n) = (2/H) sum_r h_rj S_rn over the encoding rows r of \
                 a cycle, S_rn readout n after row r's preparation; outputs readout-major");
        }
        Some(block)
    }

    fn resolved_bool(v: (bool, crate::protocol::Source)) -> Value {
        json!({ "Value": v.0, "Source": v.1.as_str() })
    }

    /// The raw series of a Hadamard protocol under `sourcedata/sub-X/[ses-Y/]perf/` (P6 part A,
    /// "Outputs"): the raw volumes per echo, their sidecar, one row per raw volume, and the raw
    /// truth with the per-preparation factor table.
    fn write_hadamard_sourcedata(root: &Path, names: &Names, p: &Protocol, out: &SeriesOutput,
                                 h: &crate::series::HadamardSeries) -> Result<(), String> {
        let spec = p.hadamard.as_ref().ok_or("a Hadamard series without [hadamard]")?;
        let perf = root.join("sourcedata").join(names.perf_dir());
        let gt_dir = perf.join("ground-truth");
        std::fs::create_dir_all(&gt_dir).map_err(|e| format!("{}: {e}", gt_dir.display()))?;
        let prefix_s = perf.join(names.stem()).to_string_lossy().to_string();
        let sched = &h.schedule;
        let n_echo = p.echo_times_s.len();
        for e in 0..n_echo {
            let echo_tag = if n_echo > 1 { format!("_echo-{}", e + 1) } else { String::new() };
            let prefix_e = format!("{prefix_s}{echo_tag}");
            let (mag, phase) = if e == 0 { (&h.raw_mag, &h.raw_phase) } else { (&h.raw_more_echoes[e - 1].0, &h.raw_more_echoes[e - 1].1) };
            let info = SidecarInfo {
                manufacturer: "aslscan".to_string(),
                phase_encoding_direction: p.phase_encoding_direction.clone(),
                total_readout_time: p.total_readout_time_s,
                echo_time: p.echo_times_s[e],
                partial_fourier: out.acquisition.partial_fourier,
                accel: out.acquisition.accel,
                mb: p.mb,
                repetition_time_s: sched.raw_rows.iter().find(|r| r.kind != crate::rows::RowKind::M0scan).map(|r| r.tr),
                b0_field_source: None,
            };
            write_complex_4d(&prefix_e, "asl", out.acq_grid.dims, h.n_raw, mag, phase, &out.acq_grid, &info)
                .map_err(|e| e.to_string())?;
            let side = json!({
                "Description": "the raw (encoded) series of a Hadamard time-encoded acquisition, before decoding; \
                                the dataset's decoded series is the main one",
                "EchoTime": p.echo_times_s[e],
                "RepetitionTimePreparation": sched.raw_rows.iter().map(|r| r.tr).collect::<Vec<f64>>(),
                "RawVolumes": h.n_raw,
                "RawVolumeTable": format!("{}_rawvolumes.tsv", names.stem()),
                "AslscanSimulation": { "Hadamard": hadamard_block(p, out, e) },
            });
            write_json(&PathBuf::from(format!("{prefix_e}_part-mag_asl.json")), &side)?;
            let mut ph_side = side.clone();
            ph_side["Units"] = json!("rad");
            write_json(&PathBuf::from(format!("{prefix_e}_part-phase_asl.json")), &ph_side)?;
        }
        // P7 part B: under Look-Locker each raw volume is one readout of its preparation
        let ll = spec.readouts > 1;
        let mut tsv = String::from("raw_volume\tkind\tcycle\tencoding_row\tlabeled_subboli\tfirst_preparation\tpreparations");
        tsv.push_str(if ll { "\treadout\n" } else { "\n" });
        for (r, rv) in sched.raws.iter().enumerate() {
            tsv.push_str(&format!("{r}\t{}\t{}\t{}\t{}\t{}\t{}",
                if rv.encoding_row.is_some() { "encoded" } else { sched.raw_rows[r].kind.as_str() },
                rv.cycle.map_or("n/a".to_string(), |c| (c + 1).to_string()),
                rv.encoding_row.map_or("n/a".to_string(), |i| i.to_string()),
                labeled(spec.order, rv.encoding_row).iter().map(|j| j.to_string()).collect::<Vec<_>>().join(","),
                rv.prep, rv.n_preps));
            if ll {
                tsv.push_str(&format!("\t{}", rv.readout + 1));
            }
            tsv.push('\n');
        }
        std::fs::write(format!("{prefix_s}_rawvolumes.tsv"), tsv).map_err(|e| e.to_string())?;

        // the raw truth
        let gt_prefix = gt_dir.join(names.stem()).to_string_lossy().to_string();
        let mean = "volume-weighted mean over the phantom voxels each acquisition voxel overlaps";
        let moved = h.raw_delta_m_static.is_some();
        write_4d(&PathBuf::from(format!("{gt_prefix}_desc-deltam_gt.nii.gz")), out.acq_grid.dims, h.n_raw, &h.raw_delta_m, &out.acq_grid)
            .map_err(|e| e.to_string())?;
        write_json(&PathBuf::from(format!("{gt_prefix}_desc-deltam_gt.json")), &json!({
            "Units": "arbitrary (same as M0map)",
            "Description": "+ the encoded kinetic sum of each raw volume's labeled sub-boli (no suppression or \
                            physiological factor), today's conventions; zero for m0scan raw volumes",
            "Frame": if out.readout.is_some() { "static (a 3D raw volume's shots have their own poses)" }
                     else if moved { "moved by the raw volume's pose" } else { "static" },
            "Resampling": mean,
        }))?;
        if let Some(gts) = &h.raw_delta_m_static {
            write_4d(&PathBuf::from(format!("{gt_prefix}_desc-deltamStatic_gt.nii.gz")), out.acq_grid.dims, h.n_raw, gts, &out.acq_grid)
                .map_err(|e| e.to_string())?;
        }
        for (desc, data) in [("deltamIntravascular", &h.raw_delta_m_iv), ("deltamSuppressed", &h.raw_delta_m_suppressed),
                             ("deltamArterial", &h.raw_delta_m_arterial)] {
            if let Some(d) = data {
                write_4d(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.nii.gz")), out.acq_grid.dims, h.n_raw, d, &out.acq_grid)
                    .map_err(|e| e.to_string())?;
                write_json(&PathBuf::from(format!("{gt_prefix}_desc-{desc}_gt.json")), &json!({
                    "Units": "arbitrary (same as M0map)", "Description": "per raw volume, the encoded sum of its labeled sub-boli's part (P4's definition per row)", "Resampling": mean,
                }))?;
            }
        }
        write_volume_tables(&gt_prefix, p, out)?;
        let mut tsv = String::from(
            "preparation\traw_volume\tshot\tencoding_row\tlabeled_subboli\tstart\tlabeling_window_start\tlabeling_window_end\tlabel_factor\ttissue_factor\tsuppression_factor\tshot_gain\n");
        for (i, f) in h.prep_factors.iter().enumerate() {
            tsv.push_str(&format!("{i}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", f.raw, f.shot,
                f.encoding_row.map_or("n/a".to_string(), |r| r.to_string()),
                labeled(spec.order, f.encoding_row).iter().map(|j| j.to_string()).collect::<Vec<_>>().join(","),
                f.start_s, f.labeling_window[0], f.labeling_window[1], f.label,
                if f.tissue.is_nan() { "per slice (desc-physio_gt.tsv)".to_string() } else { f.tissue.to_string() },
                f.suppression.map_or("n/a".to_string(), |x| x.to_string()), f.shot_gain));
        }
        std::fs::write(format!("{gt_prefix}_desc-preparations_gt.tsv"), tsv).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// The spiral path's certified-bound target (P5 part C); spirals need the `kspace` feature, and
    /// `resolve_readout` refuses them without it.
    #[cfg(feature = "kspace")]
    fn spiral_bound_target() -> f64 {
        mrsim_acq::tseg::BOUND_TARGET
    }
    #[cfg(not(feature = "kspace"))]
    fn spiral_bound_target() -> f64 {
        unreachable!("spiral readouts are refused without the kspace feature")
    }

    /// The spiral reconstruction's fixed parameters, as the sidecar records them (P5 part C).
    #[cfg(feature = "kspace")]
    fn spiral_reconstruction_block() -> Value {
        use mrsim_acq::grid_recon as gr;
        json!({
            "Method": "density-weighted least squares per coil and partition on the image band, a fixed Chebyshev \
                       semi-iteration (linear in the data) on Q A^H W A Q x = Q A^H W d, then the Roemer combine",
            "Band": if gr::BAND_LIMITED {
                "Q projects onto the images whose DFT vanishes outside the disc of the largest sample radius \
                 (frequency i up to (n-1)/2, i - n above)"
            } else {
                "none: the full n x n grid"
            },
            "Iterations": gr::LS_ITERATIONS, "IntervalRatio": gr::LS_KAPPA,
            "PowerIterations": gr::POWER_ITERATIONS, "EigenvalueMargin": gr::LAMBDA_MARGIN,
            "DensityCompensation": "Pipe-Menon, operator form, normalized so a constant object grids to its Cartesian value \
                                    at the image centre",
            "DensityIterations": gr::DCF_ITERATIONS,
        })
    }
    #[cfg(not(feature = "kspace"))]
    fn spiral_reconstruction_block() -> Value {
        unreachable!("spiral readouts are refused without the kspace feature")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rows::RowKind;

    #[test]
    fn names_follow_bids() {
        let n = Names::new("01", Some("02"));
        assert_eq!(n.stem(), "sub-01_ses-02");
        assert_eq!(n.perf_dir(), "sub-01/ses-02/perf");
        assert_eq!(n.rel("_part-mag_asl.nii.gz"), "sub-01/ses-02/perf/sub-01_ses-02_part-mag_asl.nii.gz");
        let n = Names::new("sub-01", None);
        assert_eq!(n.stem(), "sub-01");
        assert_eq!(n.perf_dir(), "sub-01/perf");
        assert_eq!(ground_truth_tail("att"), "_desc-att_gt.nii.gz");
    }

    #[test]
    fn aslcontext_is_written_in_row_order() {
        let rows = vec![
            Row { kind: RowKind::M0scan, t: 0.0, tau: 0.0, tr: 8.0 },
            Row { kind: RowKind::Control, t: 3.6, tau: 1.8, tr: 4.0 },
            Row { kind: RowKind::Label, t: 3.6, tau: 1.8, tr: 4.0 },
            Row { kind: RowKind::Deltam, t: 3.6, tau: 1.8, tr: 4.0 },
        ];
        assert_eq!(aslcontext_tsv(&rows), "volume_type\nm0scan\ncontrol\nlabel\ndeltam\n");
    }
}
