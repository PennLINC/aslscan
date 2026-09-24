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
    use crate::protocol::{M0Type, Protocol};
    use crate::series::SeriesOutput;

    fn write_json(path: &Path, v: &Value) -> Result<(), String> {
        let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    fn resolved(v: (f64, crate::protocol::Source)) -> Value {
        json!({ "Value": v.0, "Source": v.1.as_str() })
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
        json!({
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
        })
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
        let info = SidecarInfo {
            phase_encoding_direction: p.phase_encoding_direction.clone(),
            total_readout_time: p.total_readout_time_s,
            echo_time: p.echo_time_s,
            partial_fourier: out.acquisition.partial_fourier,
            accel: out.acquisition.accel,
            mb: p.mb,
            b0_field_source: None,
        };
        write_complex_4d(&prefix_s, "asl", out.acq_grid.dims, out.n_volumes, &out.mag, &out.phase, &out.acq_grid, &info)
            .map_err(|e| e.to_string())?;
        // The shared writer's part sidecars are DWI-flavoured stubs. Each part gets the complete
        // ASL sidecar (the phase part with its Units). There is deliberately NO inheritance-level
        // `_asl.json`: with `part-` entities there is no `_asl.nii.gz`, and bids-validator 3
        // flags such a file as SIDECAR_WITHOUT_DATAFILE; it also checks required keys per part
        // file without merging a less specific sidecar in, so each part must be complete.
        let mut side: Map<String, Value> = p.input_sidecar.as_object().cloned().unwrap_or_default();
        // Standard keys describe what was SIMULATED, not what the input said: an overlay may
        // have overridden the sidecar's LabelingEfficiency, and PartialFourier or the
        // acceleration factor come from the overlay/protocol, not the input. The originals are
        // kept under AslscanSimulation.InputValuesReplaced so nothing is lost.
        let mut effective: Vec<(&str, Value)> = vec![
            ("LabelingEfficiency", json!(p.alpha.0)),
            ("PartialFourier", json!(out.acquisition.partial_fourier)),
            ("ParallelReductionFactorInPlane", json!(out.acquisition.accel)),
            ("MultibandAccelerationFactor", json!(p.mb)),
            ("TotalAcquiredPairs", json!(p.total_acquired_pairs())),
        ];
        if let Some(ir) = &p.ir {
            // Standard fields; a simasl-legal negative excitation angle is written as its
            // positive equivalent (BIDS: 0..360), the signed value staying in the block above.
            effective.push(("InversionTime", json!(ir.params.inversion_time)));
            effective.push(("FlipAngle", json!(ir.params.excitation_flip_deg.rem_euclid(360.0))));
        }
        let mut replaced = Map::new();
        for (k, v) in effective {
            if let Some(old) = side.get(k) {
                // numbers compare as numbers: an input `90` is not replaced by `90.0`
                let same = match (old.as_f64(), v.as_f64()) {
                    (Some(a), Some(b)) => a == b,
                    _ => *old == v,
                };
                if !same {
                    replaced.insert(k.to_string(), old.clone());
                }
            }
            side.insert(k.to_string(), v);
        }
        let mut sim = simulation_block(p, out);
        sim["InputValuesReplaced"] = Value::Object(replaced);
        side.insert("AslscanSimulation".to_string(), sim);
        write_json(&PathBuf::from(format!("{prefix_s}_part-mag_asl.json")), &Value::Object(side.clone()))?;
        side.insert("Units".to_string(), json!("rad"));
        write_json(&PathBuf::from(format!("{prefix_s}_part-phase_asl.json")), &Value::Object(side))?;
        std::fs::write(format!("{prefix_s}_aslcontext.tsv"), aslcontext_tsv(&p.rows)).map_err(|e| e.to_string())?;

        // The separate M0 scan.
        if p.m0_type == M0Type::Separate {
            let (mag, _phase) = out.m0.as_ref().ok_or("M0Type Separate but no M0 volume was simulated")?;
            write_3d(&PathBuf::from(format!("{prefix_s}_m0scan.nii.gz")), out.acq_grid.dims, mag, &out.acq_grid)
                .map_err(|e| e.to_string())?;
            let mut m0side = Map::new();
            for k in ["Manufacturer", "MagneticFieldStrength", "MRAcquisitionType", "PhaseEncodingDirection",
                      "TotalReadoutTime", "EchoTime", "SliceTiming", "SliceEncodingDirection",
                      "AcquisitionVoxelSize", "FlipAngle"] {
                if let Some(v) = p.input_sidecar.get(k) {
                    m0side.insert(k.to_string(), v.clone());
                }
            }
            // The M0 scan shares the ASL readout, so its effective readout values are the same.
            m0side.insert("PartialFourier".to_string(), json!(out.acquisition.partial_fourier));
            m0side.insert("ParallelReductionFactorInPlane".to_string(), json!(out.acquisition.accel));
            m0side.insert("MultibandAccelerationFactor".to_string(), json!(p.mb));
            m0side.insert("RepetitionTimePreparation".to_string(), json!(p.m0_repetition_time_s));
            m0side.insert("IntendedFor".to_string(), json!([
                format!("bids::{}", names.rel("_part-mag_asl.nii.gz")),
                format!("bids::{}", names.rel("_part-phase_asl.nii.gz")),
            ]));
            m0side.insert("AslscanSimulation".to_string(), json!({
                "Seed": out.seeds.1, "Magnitude": true, "Contrast": "se",
                "Note": "a plain spin-echo readout at its own repetition time: no suppression, no inversion, no motion",
            }));
            write_json(&PathBuf::from(format!("{prefix_s}_m0scan.json")), &Value::Object(m0side))?;
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
            "Description": if moved {
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
            for (desc, data) in [("acqT2map", &gt.acq_t2_ms), ("acqT2primemap", &gt.acq_t2p_ms)] {
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
        Ok(())
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
