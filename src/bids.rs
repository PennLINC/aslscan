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

    /// Everything the simulator resolved, for the output sidecar.
    fn simulation_block(p: &Protocol, out: &SeriesOutput) -> Value {
        let a = &out.acquisition;
        json!({
            "Simulator": { "Name": "aslscan", "Version": env!("CARGO_PKG_VERSION") },
            "Seed": out.seeds.0,
            "M0ScanSeed": out.seeds.1,
            "T2Mode": out.mode.as_str(),
            "Labels": out.labels.iter().map(|(l, n)| json!({ "Label": l, "Name": n })).collect::<Vec<_>>(),
            "Compartments": out.n_compartments,
            "CompartmentOrder": "tissue per label, then labeled blood per label",
            "Resolved": {
                "LabelingEfficiency": resolved(p.alpha),
                "LambdaBloodBrain": resolved(p.lambda),
                "T1ArterialBlood": resolved(p.t1b),
                "T2Blood": resolved(p.t2_blood_s),
                "AcqContrast": "se",
                "M0RepetitionTime": p.m0_repetition_time_s,
            },
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
            "GeneratedBy": [{ "Name": "aslscan", "Version": env!("CARGO_PKG_VERSION"),
                              "Description": "Simulated ASL on the mrsim-acq acquisition stage" }],
        }))?;
        std::fs::write(root.join(".bidsignore"), "**/ground-truth/\n").map_err(|e| e.to_string())?;

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
        // The shared writer's part sidecars are DWI-flavoured stubs; the ASL sidecar is the
        // inheritance-level `_asl.json`, and the phase part needs only its Units.
        let _ = std::fs::remove_file(format!("{prefix_s}_part-mag_asl.json"));
        write_json(&PathBuf::from(format!("{prefix_s}_part-phase_asl.json")), &json!({ "Units": "rad" }))?;
        let mut side: Map<String, Value> = p.input_sidecar.as_object().cloned().unwrap_or_default();
        side.insert("AslscanSimulation".to_string(), simulation_block(p, out));
        write_json(&PathBuf::from(format!("{prefix_s}_asl.json")), &Value::Object(side))?;
        std::fs::write(format!("{prefix_s}_aslcontext.tsv"), aslcontext_tsv(&p.rows)).map_err(|e| e.to_string())?;

        // The separate M0 scan.
        if p.m0_type == M0Type::Separate {
            let (mag, _phase) = out.m0.as_ref().ok_or("M0Type Separate but no M0 volume was simulated")?;
            write_3d(&PathBuf::from(format!("{prefix_s}_m0scan.nii.gz")), out.acq_grid.dims, mag, &out.acq_grid)
                .map_err(|e| e.to_string())?;
            let mut m0side = Map::new();
            for k in ["Manufacturer", "MagneticFieldStrength", "MRAcquisitionType", "PhaseEncodingDirection",
                      "TotalReadoutTime", "EchoTime", "SliceTiming", "AcquisitionVoxelSize", "FlipAngle"] {
                if let Some(v) = p.input_sidecar.get(k) {
                    m0side.insert(k.to_string(), v.clone());
                }
            }
            m0side.insert("RepetitionTimePreparation".to_string(), json!(p.m0_repetition_time_s));
            m0side.insert("IntendedFor".to_string(), json!([
                format!("bids::{}", names.rel("_part-mag_asl.nii.gz")),
                format!("bids::{}", names.rel("_part-phase_asl.nii.gz")),
            ]));
            m0side.insert("AslscanSimulation".to_string(), json!({ "Seed": out.seeds.1, "Magnitude": true }));
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
        write_json(&PathBuf::from(format!("{gt_prefix}_desc-deltam_gt.json")), &json!({
            "Units": "arbitrary (same as M0map)",
            "Description": "+delta_m at each label/deltam row's own timing, box-averaged; zero for other rows",
            "Resampling": mean,
        }))?;
        let dseg: Vec<i16> = gt.dseg.iter().map(|l| *l as i16).collect();
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
