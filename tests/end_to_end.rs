//! End-to-end properties on the checked-in crop (features `io` + `test-hooks`): the
//! blood-compartment linearity identity with its negative controls (and with background
//! suppression or shared motion on), the noise-variance ratio of control minus label, and the
//! asl002-shaped suppression run of the P3 acceptance criteria.
#![cfg(all(feature = "io", feature = "test-hooks"))]

use std::path::Path;

use aslscan::longitudinal::{tissue_mz, Suppression};
use aslscan::mrsignal::tissue_se;
use aslscan::phantom::{self, Phantom, T2Mode};
use aslscan::protocol::{parse, Overlay, Protocol};
use aslscan::series::{complex_from, simulate_with, RowOverride, SeriesOutput};
use mrsim_acq::phase::PhaseModel;
use nifti::{IntoNdArray, NiftiObject};
use serde_json::{json, Value};

fn crop() -> Phantom {
    phantom::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap()
}

/// PCASL on the crop at 2 mm in-plane, 3 mm slices (12 x 12 x 2), oversample 2, no noise, no
/// GRAPPA, no spikes: the settings the linearity property requires. `suppression` adds two
/// pulses at 2.0 and 3.2 s (t = 3.6) at the default efficiency; `extra` is appended to the
/// overlay.
fn protocol_with(rows: &str, noise: f64, suppression: bool, extra: &str) -> Protocol {
    let mut s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": suppression, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012
    });
    if suppression {
        s["BackgroundSuppressionNumberPulses"] = json!(2);
        s["BackgroundSuppressionPulseTime"] = json!([2.0, 3.2]);
    }
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nnoise_variance = {noise}\nsignal_scale = 100.0\npartial_fourier = 0.75\n{extra}"
    )).unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

fn protocol(rows: &str, noise: f64) -> Protocol {
    protocol_with(rows, noise, false, "")
}

fn phase() -> PhaseModel {
    PhaseModel { global: 0.0, background: Default::default(), prep: None }
}

/// Complex image of the single volume of a one-row series.
fn image_of(p: &Protocol, ov: RowOverride) -> Vec<(f64, f64)> {
    let out = simulate_with(p, &crop(), T2Mode::Auto, &phase(), ov).unwrap();
    assert_eq!(out.n_volumes, 1);
    complex_from(&out.mag, &out.phase)
}

fn image(rows: &str, ov: RowOverride) -> Vec<(f64, f64)> {
    image_of(&protocol(rows, 0.0), ov)
}

/// max |I_C - I_L - I_B| against the spec's tolerance, as a ratio (1 = at the tolerance).
fn residual_of(c: &[(f64, f64)], l: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let max_b = b.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
    let max_c = c.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
    assert!(max_b > 0.0 && max_c > 0.0, "empty images");
    let atol = 1e-6 * max_b.max(max_c);
    let rtol = 1e-5;
    let mut worst = 0.0f64;
    for i in 0..c.len() {
        let res = (c[i].0 - l[i].0 - b[i].0).hypot(c[i].1 - l[i].1 - b[i].1);
        let bound = atol + rtol * b[i].0.hypot(b[i].1);
        worst = worst.max(res / bound);
    }
    worst
}

fn linearity_residual(ov_c: RowOverride, ov_l: RowOverride, ov_b: RowOverride) -> f64 {
    residual_of(&image("control", ov_c), &image("label", ov_l), &image("deltam", ov_b))
}

#[test]
fn blood_compartment_linearity_holds() {
    let worst = linearity_residual(RowOverride::None, RowOverride::None, RowOverride::None);
    println!("linearity: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "I_C - I_L != I_B: worst residual is {worst:.2}x the tolerance");
}

#[test]
fn linearity_fails_for_a_flipped_label_sign() {
    let worst = linearity_residual(RowOverride::None, RowOverride::FlipLabelSign, RowOverride::None);
    assert!(worst > 1e3, "a flipped label sign must break the identity by far more than the tolerance: {worst}");
}

#[test]
fn linearity_fails_for_a_control_label_swap() {
    let worst = linearity_residual(RowOverride::SwapControlLabel, RowOverride::SwapControlLabel, RowOverride::None);
    assert!(worst > 1e3, "swapped control/label must break the identity: {worst}");
}

#[test]
fn linearity_fails_when_label_rows_wire_blood_into_the_wrong_compartment() {
    // In the label run the blood decays at tissue T2/T2' (compartment 0) while the deltam run
    // keeps it in its own compartment: the per-line readout weighting differs, so the identity
    // fails. NOTE a mis-wiring applied to every run alike would NOT fail: I_L and I_B would carry
    // the same mis-wired term and still subtract to it. The identity catches row-dependent
    // wiring errors; consistent ones are what the class/voxel cross-check and the ground-truth
    // delta_m are for.
    let worst = linearity_residual(RowOverride::None, RowOverride::BloodIntoTissue0, RowOverride::BloodIntoTissue0);
    assert!(worst > 1e2, "label-row blood wired into tissue compartment 0 must break the identity: {worst}");
}

#[test]
fn linearity_holds_with_background_suppression_on() {
    // The three runs share the pulses, so the suppressed tissue cancels in C - L and the blood
    // carries the same label factor (0.81) in L and B.
    let run = |rows: &str| image_of(&protocol_with(rows, 0.0, true, ""), RowOverride::None);
    let worst = residual_of(&run("control"), &run("label"), &run("deltam"));
    println!("linearity with suppression: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn linearity_holds_with_a_shared_pose() {
    // Motion is linear per compartment, and the same pose in all three runs keeps the identity.
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let tsv = dir.join("pose.tsv");
    std::fs::write(&tsv, "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n1.3\t-0.7\t0.0\t0.0\t0.0\t0.05\n").unwrap();
    let path = tsv.to_string_lossy().replace('\\', "\\\\");
    let extra = format!("[motion]\nmode = \"trajectory\"\ntrajectory = \"{path}\"\n");
    let run = |rows: &str| image_of(&protocol_with(rows, 0.0, false, &extra), RowOverride::None);
    let worst = residual_of(&run("control"), &run("label"), &run("deltam"));
    println!("linearity with a shared pose: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn linearity_holds_under_compat() {
    // Compat scales tissue and blood of a voxel by the same exp(-TE/T2) before the acquisition
    // and turns the readout's relaxation off: still linear per compartment. 2 mm voxel-centre
    // grid, so the offset box weights are exercised too.
    let run = |rows: &str| {
        let s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.01, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.0], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.001
        });
        let ov: Overlay = toml::from_str("[compat]\nasldro = true\n").unwrap();
        let p = parse(&s, &format!("volume_type\n{rows}\n"), Some(&ov), crop().params.as_ref()).unwrap();
        assert!(p.compat.is_some());
        image_of(&p, RowOverride::None)
    };
    let worst = residual_of(&run("control"), &run("label"), &run("deltam"));
    println!("linearity under compat: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
}

/// Benchmark A (and E) of the P2 addendum on the crop, through the Python driver and simasl
/// itself. Runs only when `ASLSCAN_SIMASL_ENV` names the micromamba environment with `asldro`
/// and the binary was built (`--features cli`, so `CARGO_BIN_EXE_aslscan` exists); otherwise it
/// says loudly that it skipped. `ASLSCAN_MICROMAMBA` overrides the micromamba executable.
#[test]
fn compat_benchmark_a_on_the_crop() {
    let Ok(env) = std::env::var("ASLSCAN_SIMASL_ENV") else {
        eprintln!("SKIPPED compat_benchmark_a_on_the_crop: set ASLSCAN_SIMASL_ENV to the simasl micromamba environment");
        return;
    };
    let Some(bin) = option_env!("CARGO_BIN_EXE_aslscan") else {
        eprintln!("SKIPPED compat_benchmark_a_on_the_crop: the aslscan binary is not built (add --features cli)");
        return;
    };
    let mm = std::env::var("ASLSCAN_MICROMAMBA").unwrap_or_else(|_| "micromamba".to_string());
    let driver = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/compat_asldro.py");
    for bench in ["A", "E"] {
        let out = std::process::Command::new(&mm)
            .args(["run", "-n", &env, "python", driver, bench, "--crop", "--aslscan", bin])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("could not run micromamba");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        println!("{text}");
        assert!(out.status.success(), "benchmark {bench} on the crop failed:\n{text}");
    }
}

/// A PCASL protocol on the crop with every P4 part on: two slab suppression pulses (one during
/// labeling), crushing alternating 0 / 4 cm/s, exchange, per-label arterial values whose windows
/// contain the readout, and physiological noise.
fn p4_all(rows: &str, extra: &str) -> Protocol {
    let n = rows.split(',').count();
    let venc: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 4.0 }).collect();
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2,
        "BackgroundSuppressionPulseTime": [1.5, 3.2], "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012, "VascularCrushing": true, "VascularCrushingVENC": venc
    });
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n\
         [background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n\
         [kinetic]\nexchange_time = 0.4\n\
         [macrovascular]\narterial_blood_volume = {{ grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }}\n\
         arterial_transit_time = {{ grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }}\n\
         [vascular_crushing]\narterial_velocity = {{ grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }}\n\
         [physio]\ntissue_cardiac = 0.02\ntissue_respiratory = 0.01\nlabel_cardiac = 0.03\nlabel_drift = 0.01\n{extra}"
    )).unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

/// One-row P4 series for the linearity identity: every part on, crushing at 4 cm/s, the
/// exchange time given. One-row series share the physiological factors (row 0 at t = 0).
fn p4_one(row: &str, tau_ex: f64) -> Protocol {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2,
        "BackgroundSuppressionPulseTime": [1.5, 3.2], "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012, "VascularCrushing": true, "VascularCrushingVENC": 4.0
    });
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n\
         [background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n\
         [kinetic]\nexchange_time = {tau_ex}\n\
         [macrovascular]\narterial_blood_volume = {{ grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }}\n\
         arterial_transit_time = {{ grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }}\n\
         [vascular_crushing]\narterial_velocity = {{ grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }}\n\
         [physio]\ntissue_cardiac = 0.02\ntissue_respiratory = 0.01\nlabel_cardiac = 0.03\nlabel_drift = 0.01\n"
    )).unwrap();
    parse(&s, &format!("volume_type\n{row}\n"), Some(&ov), crop().params.as_ref()).unwrap()
}

fn p4_linearity(tau_ex: f64, ov_c: RowOverride, ov_l: RowOverride) -> f64 {
    let c = image_of(&p4_one("control", tau_ex), ov_c);
    let l = image_of(&p4_one("label", tau_ex), ov_l);
    let b = image_of(&p4_one("deltam", tau_ex), RowOverride::None);
    residual_of(&c, &l, &b)
}

#[test]
fn linearity_holds_with_every_p4_part_on() {
    // the arterial compartment is in its window at the readout (t = 3.6, 3.65; aATT 2.5 / 2.7)
    let out = simulate_with(&p4_one("label", 0.4), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert!(out.ground_truth.delta_m_arterial.as_ref().unwrap().iter().any(|v| *v > 0.0));
    let worst = p4_linearity(0.4, RowOverride::None, RowOverride::None);
    println!("linearity with every P4 part on: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn linearity_fails_when_the_extravascular_part_lands_in_the_blood() {
    let worst = p4_linearity(0.4, RowOverride::None, RowOverride::ExtravascularIntoBlood);
    // measured 54x the tolerance (the identity itself holds at 0.18): the exchanged label is a
    // smaller part than the whole bolus the P1 controls move, so the margin is smaller too
    println!("extravascular part in the blood: residual / tolerance = {worst:.1}");
    assert!(worst > 10.0, "the extravascular part in the blood compartment must break the identity: {worst}");
}

#[test]
fn linearity_fails_for_a_control_label_swap_on_the_p4_path() {
    // the swap gives the control row the label: the P4 label path must run for it
    let worst = p4_linearity(0.4, RowOverride::SwapControlLabel, RowOverride::SwapControlLabel);
    assert!(worst > 1e2, "swapped control/label must break the identity on the P4 path: {worst}");
}

#[test]
fn linearity_fails_when_the_intravascular_part_lands_in_tissue_0() {
    // a slow exchange keeps a measurable intravascular part for the control to move
    let worst = p4_linearity(10.0, RowOverride::None, RowOverride::BloodIntoTissue0);
    assert!(worst > 1e2, "the intravascular part in tissue compartment 0 must break the identity: {worst}");
}

#[test]
fn p4_sidecar_blocks_and_ground_truth_files() {
    let p = p4_all("control,label,control,label", "");
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-p4-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
    let sim = &side["AslscanSimulation"];
    assert_eq!(sim["Exchange"]["ExchangeTime"], json!(0.4));
    assert_eq!(sim["Macrovascular"]["T2Arterial"]["Source"], json!("T2Blood"));
    assert_eq!(sim["Macrovascular"]["ArterialBloodVolume"]["grey_matter"], json!(0.03));
    assert!(sim["CompartmentOrder"].as_str().unwrap().contains("arterial"));
    assert_eq!(sim["VascularCrushing"]["Venc"], json!([0.0, 4.0, 0.0, 4.0]));
    let surv = sim["VascularCrushing"]["Survival"].as_array().unwrap();
    assert_eq!(surv.len(), 4);
    assert_eq!(surv[0][0], json!(1.0));
    assert!(surv[1][0].as_f64().unwrap() < 1.0);
    assert_eq!(sim["BackgroundSuppressionModel"], json!("bolus-position"));
    assert_eq!(sim["BackgroundSuppression"]["PulseRegion"], json!("slab"));
    assert_eq!(sim["BackgroundSuppression"]["SlabEntryTime"], json!(0.3));
    assert!(sim.get("BackgroundSuppressionLabelFactor").is_none());
    assert_eq!(sim["Physio"]["SeedSalt"], json!("0x50485953494f"));
    // the standard fields are echoed as given
    assert_eq!(side["VascularCrushing"], json!(true));
    assert_eq!(side["VascularCrushingVENC"], json!([0.0, 4.0, 0.0, 4.0]));
    let gt = dir.join("sub-01/perf/ground-truth");
    for f in ["deltamIntravascular", "deltamSuppressed", "deltamArterial", "aBV", "aATT"] {
        assert!(gt.join(format!("sub-01_desc-{f}_gt.nii.gz")).exists(), "{f}");
        assert!(gt.join(format!("sub-01_desc-{f}_gt.json")).exists(), "{f}");
    }
    // read back: shape, and every value where the layout puts it (memory is voxel-major with
    // the volume innermost; NIfTI is x fastest with the volume outermost)
    let [nx, ny, nz] = out.acq_grid.dims;
    let n = out.n_volumes;
    let g = &out.ground_truth;
    for (f, data) in [("deltamIntravascular", &g.delta_m_iv), ("deltamSuppressed", &g.delta_m_suppressed),
                      ("deltamArterial", &g.delta_m_arterial)] {
        let data = data.as_ref().unwrap();
        assert_eq!(data.len(), nx * ny * nz * n);
        let obj = nifti::ReaderOptions::new().read_file(gt.join(format!("sub-01_desc-{f}_gt.nii.gz"))).unwrap();
        let arr = obj.into_volume().into_ndarray::<f32>().unwrap();
        assert_eq!(arr.shape(), &[nx, ny, nz, n], "{f}");
        let mut nonzero = 0;
        for ((x, y, z), v) in (0..nz).flat_map(|z| (0..ny).flat_map(move |y| (0..nx).map(move |x| (x, y, z))))
            .flat_map(|p| (0..n).map(move |v| (p, v)))
        {
            let want = data[(x + nx * (y + ny * z)) * n + v];
            assert!(want.is_finite(), "{f}");
            assert_eq!(arr[[x, y, z, v]].to_bits(), want.to_bits(), "{f} at {x},{y},{z} volume {v}");
            nonzero += usize::from(want != 0.0);
        }
        assert!(nonzero > 0, "{f} is all zero");
    }
    for (f, data) in [("aBV", &g.abv), ("aATT", &g.aatt)] {
        let data = data.as_ref().unwrap();
        let (vol, grid) = mrsim_acq::io::load_volume(&gt.join(format!("sub-01_desc-{f}_gt.nii.gz"))).unwrap();
        assert_eq!(grid.dims, [nx, ny, nz], "{f}");
        assert_eq!(&vol, data, "{f}");
        assert!(vol.iter().any(|x| *x > 0.0), "{f} is all zero");
    }
    let tsv = std::fs::read_to_string(gt.join("sub-01_desc-physio_gt.tsv")).unwrap();
    assert_eq!(tsv.lines().count(), 1 + 4 * 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compat_sidecar_names_every_pinned_value() {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Included", "RepetitionTimePreparation": [10.0, 5.0, 5.0],
        "EchoTime": 0.01, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.0], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.001
    });
    let ov: Overlay = toml::from_str("[compat]\nasldro = true\ndesired_snr = 50.0\n[signal]\nacq_contrast = \"ir\"\n").unwrap();
    let p = parse(&s, "volume_type\nm0scan\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-compat-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
    let sim = &side["AslscanSimulation"];
    let c = &sim["Compat"];
    assert_eq!(c["Asldro"], json!(true));
    for (k, v) in aslscan::protocol::COMPAT_PINNED {
        assert_eq!(c["Pinned"][k].as_f64(), Some(v), "{k}");
    }
    assert_eq!(c["Pinned"]["window"], json!("none"));
    assert_eq!(c["Pinned"]["readout_relaxation"], json!(false));
    assert_eq!(c["DesiredSnr"], json!(50.0));
    let f = out.compat.as_ref().unwrap();
    assert_eq!(c["NoiseVariance"].as_f64(), Some(f.noise_variance));
    assert_eq!(c["M0ReferenceMean"].as_f64(), Some(f.m0_reference_mean));
    assert_eq!(c["GridOrigin"], json!("voxel-centre"));
    assert_eq!(sim["Grid"]["Origin"], json!("voxel-centre"));
    assert_eq!(sim["M0ScanContrast"], json!("ir"));
    assert_eq!(sim["Resolved"]["T2Blood"]["Used"], json!(false));
    assert_eq!(sim["Acquisition"]["NoiseVariance"].as_f64(), Some(f.noise_variance));
    assert_eq!(side["ParallelReductionFactorInPlane"], json!(1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn control_minus_label_variance_is_twice_one_volumes_noise() {
    // Independent noise: Var(C - L) = 2 Var(n). A shared realization would give zero.
    let noise = 4.0;
    let out = simulate_with(&protocol("control,label,control,label", noise), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let clean = simulate_with(&protocol("control,label,control,label", 0.0), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let n = out.n_volumes;
    let nvox = out.acq_grid.dims.iter().product::<usize>();
    let (zn, zc) = (complex_from(&out.mag, &out.phase), complex_from(&clean.mag, &clean.phase));
    let noise_of = |v: usize| -> Vec<f64> { (0..nvox).map(|vox| zn[vox * n + v].0 - zc[vox * n + v].0).collect() };
    let var = |x: &[f64]| x.iter().map(|a| a * a).sum::<f64>() / x.len() as f64;
    let (n0, n1) = (noise_of(0), noise_of(1));
    let diff: Vec<f64> = n0.iter().zip(&n1).map(|(a, b)| a - b).collect();
    let ratio = var(&diff) / (0.5 * (var(&n0) + var(&n1)));
    println!("Var(C - L) / Var(one volume) = {ratio:.3}");
    assert!((1.6..=2.4).contains(&ratio), "expected about 2 for independent noise, got {ratio}");
}

// ---------------------------------------------------------------- P3 acceptance, asl002-shaped

/// asl002's real sidecar and aslcontext on the crop: 1 mm voxels (the crop's own grid, so
/// interior GM/WM voxels exist), its first six slice times, oversample 1, no noise.
fn asl002(suppression: Option<bool>, efficiency: Option<f64>) -> (Protocol, SeriesOutput) {
    let d = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/asl002/");
    let mut s: Value = serde_json::from_str(&std::fs::read_to_string(format!("{d}asl.json")).unwrap()).unwrap();
    let ctx = std::fs::read_to_string(format!("{d}aslcontext.tsv")).unwrap();
    s["AcquisitionVoxelSize"] = json!([1.0, 1.0, 1.0]);
    let timing: Vec<f64> = s["SliceTiming"].as_array().unwrap().iter().take(6).map(|v| v.as_f64().unwrap()).collect();
    s["SliceTiming"] = json!(timing);
    if let Some(b) = suppression {
        s["BackgroundSuppression"] = json!(b);
    }
    let eff = efficiency.map_or(String::new(), |e| format!("[background_suppression]\ninversion_efficiency = {e}\n"));
    let ov: Overlay = toml::from_str(&format!("[m0]\nrepetition_time = 8.0\n[acquisition]\noversample = 1\n{eff}")).unwrap();
    let p = parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap();
    let out = simulate_with(&p, &crop(), T2Mode::Class, &phase(), RowOverride::None).unwrap();
    (p, out)
}

/// Voxels of `label` whose in-plane 3x3 neighbourhood is that label, in slice `z`.
fn interior(out: &SeriesOutput, label: i32, z: usize) -> Vec<usize> {
    let [nx, ny, _] = out.acq_grid.dims;
    let d = &out.ground_truth.dseg;
    let mut v = Vec::new();
    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let all = (-1i32..=1).all(|dy| (-1i32..=1).all(|dx| {
                d[(x as i32 + dx) as usize + nx * ((y as i32 + dy) as usize + ny * z)] == label
            }));
            if all {
                v.push(x + nx * (y + ny * z));
            }
        }
    }
    v
}

#[test]
fn asl002_shaped_suppression_meets_the_acceptance_numbers() {
    let (p, dflt) = asl002(None, None);
    assert_eq!(p.rows.len(), 70);
    let f = dflt.label_factors.as_ref().unwrap();
    assert!(f.iter().all(|x| (x - 0.81).abs() < 1e-12), "default efficiency 0.95 twice: {f:?}");
    assert!(dflt.m0.is_some(), "asl002 has a separate M0 scan");

    let (_, perfect) = asl002(None, Some(1.0));
    let (_, off) = asl002(Some(false), None);
    let n = dflt.n_volumes;
    // rows 0 and 1 are the first control/label pair; slice 0 reads at 3.8 s
    let (ctrl, lab) = (0usize, 1usize);
    assert_eq!(p.rows[ctrl].kind, aslscan::rows::RowKind::Control);
    assert_eq!(p.rows[lab].kind, aslscan::rows::RowKind::Label);
    let s = Suppression::new(vec![2.05, 3.276], 1.0, false);
    let tr = p.rows[ctrl].tr;
    for (label, name, t1) in [(1, "GM", 1.33f64), (2, "WM", 0.83)] {
        let vox = interior(&perfect, label, 0);
        assert!(vox.len() >= 3, "{name}: too few interior voxels in slice 0: {}", vox.len());
        let ratio: f64 = vox.iter().map(|&v| perfect.mag[v * n + ctrl] as f64 / off.mag[v * n + ctrl] as f64).sum::<f64>() / vox.len() as f64;
        let want = tissue_mz(1.0, t1, tr, 3.8, &s) / tissue_se(1.0, t1, tr);
        println!("{name} first-slice control ratio, perfect pulses / no suppression: {ratio:.4} (closed form {want:.4})");
        assert!(ratio < 0.2, "{name}: suppression must exceed 80%: ratio {ratio}");
        assert!((ratio - want).abs() < 0.05, "{name}: ratio {ratio} vs closed form {want}");
    }
    // the complex control - label difference does not care about the tissue's suppression
    let (zp, zo) = (complex_from(&perfect.mag, &perfect.phase), complex_from(&off.mag, &off.phase));
    let nvox = perfect.acq_grid.dims.iter().product::<usize>();
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    let mut diffs = Vec::with_capacity(nvox);
    for vox in 0..nvox {
        let dp = (zp[vox * n + ctrl].0 - zp[vox * n + lab].0, zp[vox * n + ctrl].1 - zp[vox * n + lab].1);
        let doff = (zo[vox * n + ctrl].0 - zo[vox * n + lab].0, zo[vox * n + ctrl].1 - zo[vox * n + lab].1);
        scale = scale.max(doff.0.hypot(doff.1));
        diffs.push((dp.0 - doff.0).hypot(dp.1 - doff.1));
    }
    for d in diffs {
        worst = worst.max(d / scale);
    }
    // The difference is about 1% of the tissue signal, so the float32 storage of the images
    // (~1e-7 of the tissue) shows up at ~1e-5 of the difference; 1e-4 is the criterion.
    println!("complex control - label, perfect pulses vs no suppression: worst relative change {worst:.3e}");
    assert!(worst < 1e-4, "{worst}");
    // and the default-efficiency run's difference is 0.81 of it
    let zd = complex_from(&dflt.mag, &dflt.phase);
    let mut worst = 0.0f64;
    for vox in 0..nvox {
        let dd = (zd[vox * n + ctrl].0 - zd[vox * n + lab].0, zd[vox * n + ctrl].1 - zd[vox * n + lab].1);
        let doff = (zo[vox * n + ctrl].0 - zo[vox * n + lab].0, zo[vox * n + ctrl].1 - zo[vox * n + lab].1);
        worst = worst.max((dd.0 - 0.81 * doff.0).hypot(dd.1 - 0.81 * doff.1) / scale);
    }
    println!("default efficiency: worst deviation from 0.81 x the unsuppressed difference {worst:.3e}");
    assert!(worst < 1e-4, "default efficiency must scale the difference by 0.81: {worst}");
}

/// P5 part A: gradient echo at 60 degrees with suppression. The three one-row runs share the
/// preparation, so the tissue (the suppression timeline's fixed point, times sin(a)) cancels in
/// C - L and the blood carries sin(a) delta_m in L and B.
fn ge_run(rows: &str, ov: RowOverride) -> Vec<(f64, f64)> {
    image_of(&protocol_with(rows, 0.0, true, "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n"), ov)
}

#[test]
fn linearity_holds_under_gradient_echo() {
    let worst = residual_of(&ge_run("control", RowOverride::None), &ge_run("label", RowOverride::None),
                            &ge_run("deltam", RowOverride::None));
    println!("linearity under gradient echo: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn linearity_fails_under_gradient_echo_for_a_flipped_label_sign() {
    let worst = residual_of(&ge_run("control", RowOverride::None), &ge_run("label", RowOverride::FlipLabelSign),
                            &ge_run("deltam", RowOverride::None));
    assert!(worst > 1e3, "{worst}");
}

/// Rows whose preparations differ (two delays) take the propagated state: the first row is its
/// own steady state (a one-row run of it agrees), the second is not (it carries the first's
/// state, so it differs from a one-row run of itself).
#[test]
fn a_gradient_echo_series_with_differing_rows_carries_its_state() {
    let mk = |pld: serde_json::Value, rows: &str| {
        let mut s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.012
        });
        s["LabelingDuration"] = json!(1.8);
        let ov: Overlay = toml::from_str("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n\
                                          [signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 30\n").unwrap();
        let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
        parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
    };
    let two = simulate_with(&mk(json!([0.5, 1.8]), "control,control"), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert!(two.ge_rule.unwrap().contains("propagated"), "{:?}", two.ge_rule);
    let first = simulate_with(&mk(json!(0.5), "control"), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let second = simulate_with(&mk(json!(1.8), "control"), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert!(first.ge_rule.unwrap().contains("closed form"), "{:?}", first.ge_rule);
    let vol = |o: &aslscan::series::SeriesOutput, v: usize| -> Vec<f64> {
        (0..o.mag.len() / o.n_volumes).map(|i| o.mag[i * o.n_volumes + v] as f64).collect()
    };
    let (a0, a1, b, c) = (vol(&two, 0), vol(&two, 1), vol(&first, 0), vol(&second, 0));
    let peak = b.iter().fold(0.0f64, |m, x| m.max(*x));
    let d0 = a0.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let d1 = a1.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(d0 <= 1e-5 * peak, "the first row is its own steady state: {d0:e} of {peak:e}");
    assert!(d1 > 1e-3 * peak, "the second row carries the first's state: {d1:e} of {peak:e}");
}

#[test]
fn gradient_echo_sidecars_name_the_contrast() {
    let p = protocol_with("control,label", 0.0, true, "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = -30\n");
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-ge-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
    assert_eq!(side["FlipAngle"], json!(330.0));
    let sim = &side["AslscanSimulation"];
    assert_eq!(sim["Resolved"]["AcqContrast"], json!("ge"));
    assert_eq!(sim["M0ScanContrast"], json!("ge"));
    assert_eq!(sim["GradientEcho"]["ExcitationFlipAngle"]["Value"], json!(-30.0));
    assert!(sim["GradientEcho"]["SteadyState"].as_str().unwrap().contains("fixed point"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- P5 part B and D: the 3D series ----

/// A small GRASE protocol on the crop: 2 x 2 x 3 mm (12 x 12 x 2), two shots in two interleaved
/// ky segments of six 1 ms lines, refocusing at 150 degrees (stimulated echoes, so T1 enters).
fn grase(rows: &str, suppression: bool, extra: &str) -> Protocol {
    let mut s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": suppression, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "3D", "PulseSequenceType": "3Dgrase", "PhaseEncodingDirection": "j-",
        "EffectiveEchoSpacing": 0.0005, "NumberShots": 2, "FlipAngle": 150
    });
    if suppression {
        s["BackgroundSuppressionNumberPulses"] = json!(2);
        s["BackgroundSuppressionPulseTime"] = json!([2.0, 3.2]);
    }
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n{extra}"
    )).unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

/// The 2D twin of [`grase`]: the same timing with every slice excited at once (SliceTiming all
/// zero), so the kinetics and tissue the 3D series computes must be these exactly.
fn twin_2d(rows: &str, suppression: bool) -> Protocol {
    let mut s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": suppression, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.0], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.006
    });
    if suppression {
        s["BackgroundSuppressionNumberPulses"] = json!(2);
        s["BackgroundSuppressionPulseTime"] = json!([2.0, 3.2]);
    }
    let ov: Overlay = toml::from_str("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n").unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

#[test]
fn three_d_kinetics_and_tissue_are_the_2d_ones_at_zero_slice_offset() {
    for suppression in [false, true] {
        let rows = "control,label,deltam";
        let (a, _) = aslscan::series::simulate_compartments(&grase(rows, suppression, ""), &crop(), T2Mode::Auto, &phase(),
                                                             RowOverride::None).unwrap();
        let (b, _) = aslscan::series::simulate_compartments(&twin_2d(rows, suppression), &crop(), T2Mode::Auto, &phase(),
                                                             RowOverride::None).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x, y, "suppression {suppression}");
        }
    }
}

const PHYSIO_EXCHANGE: &str = "[kinetic]\nexchange_time = 0.4\n[physio]\ntissue_cardiac = 0.2\nlabel_cardiac = -0.2\nlabel_drift = 0.05\n";

fn grase_image(rows: &str, ov: RowOverride) -> Vec<(f64, f64)> {
    image_of(&grase(rows, true, PHYSIO_EXCHANGE), ov)
}

/// The linearity identity in 3D with suppression, exchange and physiological noise: the three
/// one-row runs share their shots' factors (same clock), the tissue's on the tissue group and the
/// label's on the blood and extravascular-label groups.
#[test]
fn linearity_holds_for_grase_with_physio_and_exchange() {
    let out = simulate_with(&grase("label", true, PHYSIO_EXCHANGE), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert_eq!(out.n_compartments, 9, "tissue, blood and extravascular-label groups of three labels");
    let worst = residual_of(&grase_image("control", RowOverride::None), &grase_image("label", RowOverride::None),
                            &grase_image("deltam", RowOverride::None));
    println!("linearity, GRASE with physio and exchange: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn linearity_fails_for_grase_when_the_extravascular_label_takes_the_tissue_factor() {
    let worst = residual_of(&grase_image("control", RowOverride::None), &grase_image("label", RowOverride::ExtravascularIntoTissue),
                            &grase_image("deltam", RowOverride::None));
    println!("GRASE, extravascular label in the tissue group: residual / tolerance = {worst:.1}");
    assert!(worst > 1e1, "{worst}");
}

/// The physiological processes run through the last shot of the last volume: one line per
/// (volume, shot), at each shot's excitation, and the drift at the last shots is not the clamped
/// end of the generated grid.
#[test]
fn grase_physio_is_per_shot_through_the_last_shot() {
    let p = grase("control,label,control,label", false, "[physio]\ntissue_drift = 0.05\n");
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let lines = out.physio.as_ref().unwrap();
    assert_eq!(lines.len(), 4 * 2);
    for (i, l) in lines.iter().enumerate() {
        assert_eq!((l.volume, l.slice), (i / 2, i % 2));
        let want = p.row_start[l.volume] + l.slice as f64 * 4.0 + 3.6;
        assert!((l.time - want).abs() < 1e-12, "{} vs {want}", l.time);
    }
    let last: Vec<f64> = lines.iter().rev().take(3).map(|l| l.drift).collect();
    assert!(last[0] != last[1] && last[1] != last[2], "{last:?}");
}

/// A dropout event with zero jumps in a two-shot GRASE series attenuates its shot's lines
/// (a shot gain), so the volume differs from the same series without the event, and the event is
/// recorded with its shot and no slices.
#[test]
fn grase_shot_dropout_reaches_the_acquisition() {
    let wv = "[motion]\nwithin_volume = { dropout_rate = 1.0, severity = 0.5, jump_mm = [0.0, 0.0, 0.0], jump_deg = [0.0, 0.0, 0.0] }\n";
    let with = simulate_with(&grase("control", false, wv), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let without = simulate_with(&grase("control", false, ""), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert!(!with.events.is_empty() && with.dropped.iter().all(|d| d.slices.is_empty()));
    let d = with.mag.iter().zip(&without.mag).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(d > 0.0);
}

/// P5 part B and D outputs of a GRASE series: the readout and echo-amplitude blocks, the resolved
/// standard keys, the NIfTI time step (the volume's NumberShots repetitions), the M0 sidecar, and
/// the 3D schemas of the physio and motion-event files.
#[test]
fn grase_sidecars_and_ground_truth() {
    let wv = "[physio]\ntissue_cardiac = 0.02\n[motion]\n\
              within_volume = { dropout_rate = 1.0, severity = 0.5, jump_mm = [0.3, 0.0, 0.0], jump_deg = [0.0, 0.0, 0.0] }\n\
              [m0]\nrepetition_time = 6.0\n";
    let mut p = grase("control,label", false, wv);
    p.m0_type = aslscan::protocol::M0Type::Separate;
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-grase-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let perf = dir.join("sub-01/perf");
    let side: Value = serde_json::from_str(&std::fs::read_to_string(perf.join("sub-01_part-mag_asl.json")).unwrap()).unwrap();
    let ro = &side["AslscanSimulation"]["Readout"];
    assert_eq!(ro["Type"], json!("grase"));
    assert_eq!((ro["NumberShots"].clone(), ro["KySegments"].clone(), ro["EchoTrainLength"].clone()), (json!(2), json!(2), json!(2)));
    assert!((ro["EchoSpacingMs"].as_f64().unwrap() - 12.5).abs() < 1e-9, "{}", ro["EchoSpacingMs"]);
    assert_eq!(ro["RefocusingFlipAngle"]["Value"], json!(150.0));
    assert!((ro["VolumeDuration"].as_f64().unwrap() - 8.0).abs() < 1e-12);
    let amps = &side["AslscanSimulation"]["EchoAmplitudes"]["PerEcho"];
    assert_eq!(amps["grey_matter"].as_array().unwrap().len(), 2);
    assert!(amps["blood"][0].as_f64().unwrap() > 0.0 && amps["blood"][0].as_f64().unwrap() < 1.0);
    // standard keys as resolved; no SliceTiming in 3D
    assert_eq!(side["NumberShots"], json!(2));
    assert_eq!(side["FlipAngle"], json!(150.0));
    assert!((side["EffectiveEchoSpacing"].as_f64().unwrap() - 0.0005).abs() < 1e-15);
    assert!((side["TotalReadoutTime"].as_f64().unwrap() - 0.0005 * 11.0).abs() < 1e-15);
    assert!(side.get("SliceTiming").is_none());
    // the NIfTI time step is the volume's duration
    let obj = nifti::ReaderOptions::new().read_file(perf.join("sub-01_part-mag_asl.nii.gz")).unwrap();
    assert!((obj.header().pixdim[4] - 8.0).abs() < 1e-6, "{}", obj.header().pixdim[4]);
    // the M0 sidecar: the train's refocusing angle, its readout without excitation times
    let m0: Value = serde_json::from_str(&std::fs::read_to_string(perf.join("sub-01_m0scan.json")).unwrap()).unwrap();
    assert_eq!(m0["FlipAngle"], json!(150.0));
    assert_eq!(m0["NumberShots"], json!(2));
    assert!(m0["AslscanSimulation"]["Readout"].get("ExcitationTimes").is_none());
    // physio per (volume, shot)
    let gt = perf.join("ground-truth");
    let physio = std::fs::read_to_string(gt.join("sub-01_desc-physio_gt.tsv")).unwrap();
    assert!(physio.starts_with("volume\tshot\ttime"));
    assert_eq!(physio.lines().count(), 1 + 2 * 2);
    // one motion event per volume (dropout rate 1), its shot and no slices
    let ev = std::fs::read_to_string(gt.join("sub-01_desc-motionEvents_gt.tsv")).unwrap();
    let rows: Vec<Vec<&str>> = ev.lines().skip(1).map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), out.events.len());
    for (r, e) in rows.iter().zip(&out.events) {
        assert_eq!((r[0].parse::<usize>().unwrap(), r[1].parse::<usize>().unwrap()), (e.volume, e.shot));
        assert!(e.shot < 2);
        assert_eq!(r[2], "", "no slice groups in 3D");
        assert!((r[3].parse::<f64>().unwrap() - 0.5).abs() < 1e-6);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// P5 acceptance (plan, Task 10): the linearity identity on a small GRASE protocol with every P4
/// part on, physiological noise and a shot dropout event (the three one-row runs share the seed,
/// so the same events, poses and shot factors).
fn grase_all(rows: &str, ov: RowOverride) -> Vec<(f64, f64)> {
    let mut s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2,
        "BackgroundSuppressionPulseTime": [1.5, 3.2], "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "3D", "PulseSequenceType": "3Dgrase", "PhaseEncodingDirection": "j-",
        "EffectiveEchoSpacing": 0.0005, "NumberShots": 2, "FlipAngle": 150,
        "VascularCrushing": true, "VascularCrushingVENC": 4.0
    });
    s["LabelingDuration"] = json!(1.8);
    let ov_toml: Overlay = toml::from_str(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n\
         [background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n\
         [kinetic]\nexchange_time = 0.4\n\
         [macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
         arterial_transit_time = { grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }\n\
         [vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n\
         [physio]\ntissue_cardiac = 0.05\nlabel_cardiac = 0.05\nlabel_drift = 0.02\n\
         [motion]\nwithin_volume = { dropout_rate = 1.0, severity = 0.3, jump_mm = [0.4, 0.0, 0.0], jump_deg = [0.0, 0.0, 1.0] }\n"
    ).unwrap();
    let ctx = format!("volume_type\n{rows}\n");
    image_of(&parse(&s, &ctx, Some(&ov_toml), crop().params.as_ref()).unwrap(), ov)
}

#[test]
fn linearity_holds_for_grase_with_every_part_on() {
    let worst = residual_of(&grase_all("control", RowOverride::None), &grase_all("label", RowOverride::None),
                            &grase_all("deltam", RowOverride::None));
    println!("linearity, GRASE with every P4 part, physio and a shot event: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
    let bad = residual_of(&grase_all("control", RowOverride::None), &grase_all("label", RowOverride::FlipLabelSign),
                          &grase_all("deltam", RowOverride::None));
    assert!(bad > 1e2, "{bad}");
}

// ---- P5 part C: the stack of spirals ----

/// A small spiral protocol on the crop: 2 x 2 x 3 mm (12 x 12 x 2), two interleaves of a 4 ms
/// spiral sampled every 20 us (two shots), refocusing at 150 degrees, with every P4 part,
/// physiological noise and a shot dropout event on, as [`grase_all`].
#[cfg(feature = "kspace")]
fn spiral_all(rows: &str, ov: RowOverride) -> Vec<(f64, f64)> {
    image_of(&spiral_protocol(rows), ov)
}

fn spiral_protocol(rows: &str) -> Protocol {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2,
        "BackgroundSuppressionPulseTime": [1.5, 3.2], "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "3D", "PulseSequenceType": "spiral", "FlipAngle": 150,
        "VascularCrushing": true, "VascularCrushingVENC": 4.0
    });
    let ov_toml: Overlay = toml::from_str(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n\
         [readout]\ninterleaves = 2\nspiral_readout_time = 4.0\ndwell_time = 2e-5\n\
         [background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n\
         [kinetic]\nexchange_time = 0.4\n\
         [macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
         arterial_transit_time = { grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }\n\
         [vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n\
         [physio]\ntissue_cardiac = 0.05\nlabel_cardiac = 0.05\nlabel_drift = 0.02\n\
         [motion]\nwithin_volume = { dropout_rate = 1.0, severity = 0.3, jump_mm = [0.4, 0.0, 0.0], jump_deg = [0.0, 0.0, 1.0] }\n"
    ).unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    parse(&s, &ctx, Some(&ov_toml), crop().params.as_ref()).unwrap()
}

/// P5 acceptance (plan, Task 14): the linearity identity holds through the spiral path (its
/// reconstruction is linear), with its negative control.
#[cfg(feature = "kspace")]
#[test]
fn linearity_holds_for_spirals_with_every_part_on() {
    let worst = residual_of(&spiral_all("control", RowOverride::None), &spiral_all("label", RowOverride::None),
                            &spiral_all("deltam", RowOverride::None));
    println!("linearity, spiral with every P4 part, physio and a shot event: worst residual / tolerance = {worst:.3}");
    assert!(worst <= 1.0, "{worst}");
    let bad = residual_of(&spiral_all("control", RowOverride::None), &spiral_all("label", RowOverride::FlipLabelSign),
                          &spiral_all("deltam", RowOverride::None));
    assert!(bad > 1e2, "{bad}");
}

/// The spiral sidecar records the trajectory, the certified segmentation and the reconstruction,
/// and none of the phase-encode keys.
#[cfg(feature = "kspace")]
#[test]
fn spiral_sidecars() {
    let p = spiral_protocol("control,label");
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let r3 = out.readout.as_ref().unwrap();
    assert_eq!((r3.n_shots, r3.spiral.as_ref().unwrap().interleaves), (2, 2));
    let segs = out.spiral_segmentation.as_ref().unwrap();
    assert!(!segs.is_empty() && segs.iter().all(|g| g.bound < 1e-7));
    let dir = std::env::temp_dir().join(format!("aslscan-spiral-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
    let ro = &side["AslscanSimulation"]["Readout"];
    assert_eq!(ro["Type"], json!("spiral"));
    assert_eq!(ro["Interleaves"], json!(2));
    assert_eq!(ro["Trajectory"]["SamplesPerInterleaf"], json!(200));
    assert!(ro["TimeSegmentation"]["Class"]["CertifiedBound"].as_f64().unwrap() < 1e-7);
    assert_eq!(ro["Reconstruction"]["Iterations"], json!(mrsim_acq::grid_recon::LS_ITERATIONS));
    assert!(ro["Reconstruction"]["Band"].as_str().unwrap().contains("disc of the largest sample radius"));
    assert_eq!(side["NumberShots"], json!(2));
    assert_eq!(side["DwellTime"], json!(2e-5));
    for key in ["PhaseEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing"] {
        assert!(side.get(key).is_none(), "{key}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Without the `kspace` feature a spiral is an error naming it, before anything is simulated.
#[cfg(not(feature = "kspace"))]
#[test]
fn spirals_need_the_kspace_feature() {
    let e = simulate_with(&spiral_protocol("control"), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap_err();
    assert!(e.contains("kspace"), "{e}");
}

/// Voxel mode below 180 degrees: the T1 map the echo amplitudes used is written to the ground truth
/// as `desc-acqT1map_gt`, beside the T2 and T2' maps (P5 part B).
#[test]
fn voxel_mode_grase_writes_the_acquisition_t1_map() {
    let p = grase("control,label", false, "");
    let out = simulate_with(&p, &crop(), T2Mode::Voxel, &phase(), RowOverride::None).unwrap();
    let t1 = out.ground_truth.acq_t1_ms.as_ref().expect("an acquisition T1 map at 150 degrees");
    assert_eq!(t1.len(), out.sim_grid.dims.iter().product::<usize>());
    assert!(t1.iter().any(|v| v.is_finite() && *v > 0.0));
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-acqt1-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let mut found = Vec::new();
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let path = e.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.to_string_lossy().contains("desc-acqT1map_gt") {
                found.push(path);
            }
        }
    }
    assert_eq!(found.len(), 2, "the map and its sidecar: {found:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Under gradient echo the compat sidecar names simasl's T2* factor, which the signal uses; the
/// spin-echo text is unchanged.
#[test]
fn compat_sidecar_names_the_gradient_echo_relaxation() {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Included", "RepetitionTimePreparation": [10.0, 5.0, 5.0],
        "EchoTime": 0.01, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.0], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.001
    });
    for (contrast, want) in [("ge", "exp(-EchoTime/T2*)"), ("se", "exp(-EchoTime/T2) ")] {
        let ov: Overlay = toml::from_str(&format!("[compat]\nasldro = true\n[signal]\nacq_contrast = \"{contrast}\"\n")).unwrap();
        let p = parse(&s, "volume_type\nm0scan\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
        let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        let dir = std::env::temp_dir().join(format!("aslscan-e2e-compat-{contrast}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
        let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
        let text = side["AslscanSimulation"]["Compat"]["RelaxationAtEcho"].as_str().unwrap().to_string();
        assert!(text.starts_with(want), "{contrast}: {text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// An overlay that turns a GRASE sidecar into a spiral publishes the spiral as the sequence type,
/// and both sidecars keep the input's sequence type and dwell time under `InputValuesReplaced`.
#[cfg(feature = "kspace")]
#[test]
fn spiral_override_provenance_on_both_sidecars() {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Separate", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "3D", "PulseSequenceType": "3Dgrase", "FlipAngle": 150, "DwellTime": 5e-6
    });
    let ov: Overlay = toml::from_str(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n[m0]\nrepetition_time = 6.0\n\
         [readout]\ntype = \"spiral\"\ninterleaves = 2\nspiral_readout_time = 4.0\ndwell_time = 2e-5\n").unwrap();
    let p = parse(&s, "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-spiral-ov-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let read = |f: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf").join(f)).unwrap()).unwrap() };
    let side = read("sub-01_part-mag_asl.json");
    assert_eq!((side["PulseSequenceType"].clone(), side["DwellTime"].clone()), (json!("spiral"), json!(2e-5)));
    let rep = &side["AslscanSimulation"]["InputValuesReplaced"];
    assert_eq!((rep["PulseSequenceType"].clone(), rep["DwellTime"].clone()), (json!("3Dgrase"), json!(5e-6)), "{rep}");
    let m0 = read("sub-01_m0scan.json");
    assert_eq!(m0["DwellTime"], json!(2e-5));
    let m0rep = &m0["AslscanSimulation"]["InputValuesReplaced"];
    assert_eq!(m0rep["DwellTime"], json!(5e-6), "{m0rep}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------- P6: the legacy manifest

/// A protocol of the legacy manifest: no P6 feature, and the identity schedule that states its
/// layout (one raw volume and one output per row, `NumberShots` preparations per raw volume on
/// the clock of `row_start`).
fn assert_legacy(name: &str, p: &Protocol) {
    use aslscan::schedule::{Output, Schedule};
    assert!(!p.p6_active(), "{name}: a legacy protocol must not activate the P6 path");
    let s = Schedule::identity(p);
    let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
    let n = p.rows.len();
    assert_eq!(s.raw_rows, p.rows, "{name}");
    assert_eq!((s.raws.len(), s.preps.len(), s.outputs.len()), (n, n * shots, n), "{name}");
    assert!(s.cycles.is_empty(), "{name}");
    for (v, row) in p.rows.iter().enumerate() {
        assert_eq!(s.outputs[v], Output::Raw(v), "{name}");
        assert_eq!((s.raws[v].n_preps, s.raws[v].readout, s.raws[v].cycle), (shots, 0, None), "{name}");
        for (k, prep) in s.preps_of(v).iter().enumerate() {
            assert_eq!((prep.raw, prep.shot, prep.suppression), (v, k, v), "{name}");
            assert_eq!(prep.start_s, p.row_start[v] + k as f64 * row.tr, "{name} row {v} shot {k}");
            assert_eq!(prep.venc, p.crushing.as_ref().map(|c| c.venc[v]), "{name}");
        }
    }
}

fn fixture(dir: &str) -> (Value, String) {
    let d = format!("{}/tests/fixtures/protocols/{dir}/", env!("CARGO_MANIFEST_DIR"));
    let s = serde_json::from_str(&std::fs::read_to_string(format!("{d}asl.json")).unwrap()).unwrap();
    (s, std::fs::read_to_string(format!("{d}aslcontext.tsv")).unwrap())
}

fn fixture_overlay(dir: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/protocols/{dir}/overlay.toml", env!("CARGO_MANIFEST_DIR")))
        .unwrap()
}

fn parse_named(name: &str, s: &Value, ctx: &str, ov: &str) -> Protocol {
    let ov: Overlay = toml::from_str(ov).unwrap_or_else(|e| panic!("{name}: overlay: {e}"));
    parse(s, ctx, Some(&ov), crop().params.as_ref()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// P6 Task 2: every legacy combination the gates and tests run (the regression cases of
/// `tools/regress_identity.sh` with its generated variants, the P5 acceptance fixtures, the
/// protocol builders of this file) parses with the P6 path off and has the identity layout.
/// Rejection fixtures stay in their own tests.
#[test]
fn the_legacy_manifest_keeps_the_identity_schedule() {
    // the regression cases
    let m0 = "[m0]\nrepetition_time = 8.0\n";
    let (a2, a2ctx) = fixture("asl002");
    let mut a2off = a2.clone();
    a2off["BackgroundSuppression"] = json!(false);
    for (name, s, ov) in [
        ("asl002_bs", &a2, m0.to_string()),
        ("asl002_motion", &a2, format!("seed = 5\n{m0}[motion]\nmode = \"random\"\ntrans_mm = [2.0, 2.0, 1.0]\nrot_deg = [1.0, 1.0, 2.0]\nvolumes = [5, 20, 40]\n")),
        ("asl002_noise", &a2off, format!("{m0}[acquisition]\nnoise_variance = 4.0\n")),
        ("asl002_ir", &a2off, format!("{m0}[signal]\nacq_contrast = \"ir\"\n")),
    ] {
        assert_legacy(name, &parse_named(name, s, &a2ctx, &ov));
    }
    let (mut a4, a4ctx) = fixture("asl004");
    a4["TotalReadoutTime"] = json!(0.025);
    assert_legacy("asl004_bs", &parse_named("asl004_bs", &a4, &a4ctx, m0));
    for dir in ["pasl_cutoff", "crop_pcasl", "p4_all", "p5_ge", "p5_grase", "asl003_p5"] {
        let (s, ctx) = fixture(dir);
        assert_legacy(dir, &parse_named(dir, &s, &ctx, &fixture_overlay(dir)));
    }
    let crop_json = |extra: Value| {
        let mut s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8, "M0Type": "Absent",
            "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3,
            "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05],
            "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
        });
        for (k, v) in extra.as_object().unwrap() {
            s[k] = v.clone();
        }
        s
    };
    let bs = |t: [f64; 2]| json!({"BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2, "BackgroundSuppressionPulseTime": t});
    let pasl = |extra: Value| {
        let mut s = crop_json(extra);
        let o = s.as_object_mut().unwrap();
        o.remove("LabelingDuration");
        o.insert("ArterialSpinLabelingType".into(), json!("PASL"));
        o.insert("BolusCutOffFlag".into(), json!(true));
        o.insert("BolusCutOffTechnique".into(), json!("Q2TIPS"));
        o.insert("BolusCutOffDelayTime".into(), json!(0.7));
        s
    };
    let plain = crop_json(json!({"BackgroundSuppression": false}));
    let crush = crop_json(json!({"BackgroundSuppression": false, "VascularCrushing": true, "VascularCrushingVENC": [0.0, 4.0, 0.0, 4.0]}));
    let ctx = "volume_type\ncontrol\nlabel\ncontrol\nlabel\n";
    let motion = "seed = 3\n[motion]\nmode = \"random\"\ntrans_mm = [1.0, 1.0, 0.0]\nrot_deg = [0.0, 0.0, 2.0]\nvolumes = [1, 3]\n";
    let macro_ov = "seed = 3\n[kinetic]\nexchange_time = 0.5\n[macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\narterial_transit_time = { grey_matter = 3.0, white_matter = 3.2, csf = 0.0 }\n[vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n";
    for (name, s, ov) in [
        ("crop_bs", crop_json(bs([2.0, 3.2])), "seed = 3\n"),
        ("crop_ir", plain.clone(), "seed = 3\n[signal]\nacq_contrast = \"ir\"\n"),
        ("crop_motion", plain.clone(), motion),
        ("crop_pasl_bs", pasl(bs([0.9, 1.5])), "seed = 3\n"),
        ("crop_pasl_motion", pasl(json!({"BackgroundSuppression": false})), motion),
        ("p4_physio", plain.clone(), "seed = 3\n[physio]\ntissue_cardiac = 0.02\ntissue_drift = 0.01\nlabel_respiratory = 0.03\n"),
        ("p4_macro_crush", crush, macro_ov),
        ("p4_bolus", crop_json(bs([1.5, 2.7])), "seed = 3\n[background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n"),
    ] {
        assert_legacy(name, &parse_named(name, &s, ctx, ov));
    }
    // the P5 regression variants: gradient echo with an included M0 at 90 and 35 degrees, and
    // segmented GRASE with a shot event between its shots
    let (ge, gectx) = fixture("p5_ge");
    let ge_ov = fixture_overlay("p5_ge").split("[m0]").next().unwrap().to_string();
    for flip in [90, 35] {
        let mut s = ge.clone();
        s["FlipAngle"] = json!(flip);
        s["M0Type"] = json!("Included");
        s["PostLabelingDelay"] = json!([0.0, 1.8, 1.8, 1.0, 1.0]);
        let ctx = gectx.replacen("volume_type\n", "volume_type\nm0scan\n", 1);
        assert_legacy(&format!("p5_ge{flip}"), &parse_named("p5_ge", &s, &ctx, &ge_ov));
    }
    let (gr, grctx) = fixture("p5_grase");
    let seg = format!("{}\n[motion.within_volume]\ndropout_rate = 1.0\nseverity = 0.3\njump_mm = [0.5, 0.0, 0.0]\njump_deg = [0.0, 0.0, 1.0]\n",
                      fixture_overlay("p5_grase"));
    assert_legacy("p5_grase_seg", &parse_named("p5_grase_seg", &gr, &grctx, &seg));
    // the real-geometry P5 cases: asl005 (GRASE) and asl001 (spiral, kspace only)
    let (a5, a5ctx) = fixture("asl005");
    assert_legacy("asl005_p5", &parse_named("asl005_p5", &a5, &a5ctx, &fixture_overlay("asl005_p5")));
    #[cfg(feature = "kspace")]
    {
        let (a1, a1ctx) = fixture("asl001");
        assert_legacy("asl001_p5", &parse_named("asl001_p5", &a1, &a1ctx, &fixture_overlay("asl001_p5")));
    }

    // the protocol builders of this file
    for rows in ["control", "label", "deltam", "control,label,control,label"] {
        assert_legacy("protocol_with", &protocol_with(rows, 0.0, false, ""));
        assert_legacy("protocol_with bs", &protocol_with(rows, 0.5, true, ""));
        assert_legacy("ge", &protocol_with(rows, 0.0, true, "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n"));
        assert_legacy("p4_all", &p4_all(rows, ""));
        assert_legacy("grase", &grase(rows, true, ""));
        assert_legacy("twin_2d", &twin_2d(rows, true));
        #[cfg(feature = "kspace")]
        assert_legacy("spiral", &spiral_protocol(rows));
    }
    for row in ["control", "label", "deltam"] {
        assert_legacy("p4_one", &p4_one(row, 0.4));
    }
}

// ---------------------------------------------------------------- P6 part C: multi-TE

/// PCASL on the crop, one sidecar per echo time, slices 70 ms apart (room for four echoes of a
/// 12 ms readout), no noise unless `extra` sets it; gradient echo at 60 degrees with `ge`.
fn multi_te(rows: &str, tes: &[f64], extra: &str, ge: bool) -> Protocol {
    let sidecars: Vec<Value> = tes.iter().map(|te| json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": te, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.07], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012
    })).collect();
    let contrast = if ge { "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n" } else { "" };
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n{contrast}{extra}")).unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    aslscan::protocol::parse_echoes(&sidecars, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

const SE_TES: [f64; 4] = [0.015, 0.030, 0.045, 0.060];
const GE_TES: [f64; 4] = [0.013, 0.026, 0.039, 0.052];

/// Every echo's complex image of a series.
fn echo_images(out: &SeriesOutput) -> Vec<Vec<(f64, f64)>> {
    let mut v = vec![complex_from(&out.mag, &out.phase)];
    v.extend(out.more_echoes.iter().map(|e| complex_from(&e.mag, &e.phase)));
    v
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Echo `e` of a multi-TE series is the single-echo series at `TE_e`, bit for bit, noise off
/// (both formations); with noise on, echo 1 still is (its receiver salt is zero).
#[test]
fn each_echo_is_the_single_echo_series_at_its_echo_time() {
    for (ge, tes) in [(false, &SE_TES[..3]), (true, &GE_TES[..3])] {
        for noise in [0.0, 0.5] {
            let mut p = multi_te("control,label", tes, "", ge);
            p.acq.noise_variance = noise;
            let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
            assert_eq!(out.more_echoes.len(), tes.len() - 1);
            // with noise, only echo 1 shares its receiver noise with the single-echo series
            let n_cmp = if noise == 0.0 { tes.len() } else { 1 };
            for (e, &te) in tes.iter().enumerate().take(n_cmp) {
                let mut q = multi_te("control,label", &[te], "", ge);
                q.acq.noise_variance = noise;
                assert!(!q.p6_active());
                let one = simulate_with(&q, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
                let (m, ph) = if e == 0 { (&out.mag, &out.phase) } else { (&out.more_echoes[e - 1].mag, &out.more_echoes[e - 1].phase) };
                assert_eq!(bits(m), bits(&one.mag), "ge {ge} noise {noise} echo {e} magnitude");
                assert_eq!(bits(ph), bits(&one.phase), "ge {ge} noise {noise} echo {e} phase");
            }
        }
    }
}

/// Solve the per-voxel decomposition of the first `k` echoes into parts with known decays
/// `d[part][echo]` (relative to echo 1), predict the next echo, and return the worst prediction
/// error over the peak, with the parts recovered (summed real parts, for the ratio checks).
#[allow(clippy::needless_range_loop)] // voxel-wise across the echoes and the k x k system
fn fit_parts(s: &[Vec<(f64, f64)>], d: &[Vec<f64>]) -> (f64, Vec<f64>) {
    let k = d.len();
    let peak = s[k].iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
    let mut worst = 0.0f64;
    let mut sums = vec![0.0f64; k];
    for vox in 0..s[0].len() {
        for re_im in 0..2 {
            // k x k system A x = y, A[e][part] = d[part][e]
            let mut a: Vec<Vec<f64>> = (0..k).map(|e| (0..k).map(|q| d[q][e]).collect()).collect();
            let mut y: Vec<f64> = (0..k).map(|e| if re_im == 0 { s[e][vox].0 } else { s[e][vox].1 }).collect();
            for c in 0..k {
                let piv = (c..k).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
                a.swap(c, piv);
                y.swap(c, piv);
                for r in 0..k {
                    if r != c {
                        let f = a[r][c] / a[c][c];
                        for cc in 0..k {
                            a[r][cc] -= f * a[c][cc];
                        }
                        y[r] -= f * y[c];
                    }
                }
            }
            let x: Vec<f64> = (0..k).map(|q| y[q] / a[q][q]).collect();
            let pred: f64 = (0..k).map(|q| x[q] * d[q][k]).sum();
            let got = if re_im == 0 { s[k][vox].0 } else { s[k][vox].1 };
            worst = worst.max((pred - got).abs() / peak);
            if re_im == 0 {
                for q in 0..k {
                    sums[q] += x[q];
                }
            }
        }
    }
    (worst, sums)
}

/// The crop with one T2 and one T2* for every tissue voxel, so each compartment's echo-time decay
/// is one number.
fn uniform_crop() -> Phantom {
    let mut ph = crop();
    for i in 0..ph.dseg.len() {
        if ph.dseg[i] > 0 {
            ph.t2[i] = 0.08;
            ph.t2star[i] = 0.05;
        }
    }
    ph
}

fn decay(tes: &[f64], t2_s: f64) -> Vec<f64> {
    tes.iter().map(|te| (-(te - tes[0]) / t2_s).exp()).collect()
}

/// Spin echo, exchange on, uniform relaxation: the delta-M of a `deltam` row is the
/// intravascular part at blood T2 plus the extravascular part at tissue T2, at the object level.
/// Two echoes determine both parts per voxel and predict the third; the recovered parts' ratio
/// is the ground truth's. A label row's blood part is the deltam row's, negated; wiring the label
/// row's extravascular part into the blood (the negative control) breaks that by the
/// extravascular part.
#[test]
fn spin_echo_delta_m_is_two_compartments() {
    let ph = uniform_crop();
    let tes = &SE_TES[..3];
    let run = |row: &str, ov: RowOverride| {
        let p = multi_te(row, tes, "[kinetic]\nexchange_time = 0.5\n", false);
        let out = simulate_with(&p, &ph, T2Mode::Class, &phase(), ov).unwrap();
        (p, out)
    };
    let (p, out) = run("deltam", RowOverride::None);
    let t2b = p.t2_blood_s.0;
    let d = vec![decay(tes, t2b), decay(tes, 0.08)];
    let (worst, parts) = fit_parts(&echo_images(&out), &d);
    println!("two compartments: worst prediction error {worst:e}, parts {parts:?}");
    assert!(worst < 1e-5, "{worst}");
    // the ratio of the parts at echo 1 against the truth's (sums over the grid, which the
    // reconstruction preserves at the k-space centre)
    let gt = &out.ground_truth;
    let iv: f64 = gt.delta_m_iv.as_ref().unwrap().iter().map(|&x| x as f64).sum();
    let all: f64 = gt.delta_m.iter().map(|&x| x as f64).sum();
    let want = (all - iv) / iv * (-tes[0] / 0.08).exp() / (-tes[0] / t2b).exp();
    let got = parts[1] / parts[0];
    println!("extravascular / intravascular at echo 1: {got:.4} (truth {want:.4})");
    assert!((got - want).abs() < 0.05 * want, "{got} vs {want}");
    // a label row: the same blood part, negated (its tissue part adds the tissue magnetization)
    let blood = parts[0];
    let (_, label) = run("label", RowOverride::None);
    let (worst, lp) = fit_parts(&echo_images(&label), &d);
    assert!(worst < 1e-5, "{worst}");
    assert!((lp[0] + blood).abs() < 1e-3 * blood.abs(), "label blood part {} vs deltam {blood}", lp[0]);
    // negative control: the label row's extravascular part in the blood compartment
    let (_, bad) = run("label", RowOverride::ExtravascularIntoBlood);
    let (_, bp) = fit_parts(&echo_images(&bad), &d);
    assert!((bp[0] + blood).abs() > 5.0 * blood.abs(), "mis-wired blood part {} vs deltam {blood}", bp[0]);
}

/// Gradient echo, uniform relaxation, no fieldmap: each part decays with its own T2*; the
/// arterial compartment adds a third part at arterial T2. Three echoes determine the parts and
/// predict the fourth.
#[test]
fn gradient_echo_delta_m_with_the_arterial_term() {
    let ph = uniform_crop();
    let tes = &GE_TES[..];
    // (the overlay continues the [signal] table: the arterial T2 away from the blood's 0.165 s,
    // or the two parts' decays coincide)
    let macro_ov = "t2_arterial = 0.25\n[kinetic]\nexchange_time = 0.5\n\
                    [macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
                    arterial_transit_time = { grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }\n";
    let p = multi_te("deltam", tes, macro_ov, true);
    let out = simulate_with(&p, &ph, T2Mode::Class, &phase(), RowOverride::None).unwrap();
    // T2' of the tissue labels: 1/T2' = 1/T2* - 1/T2, which the blood and arterial parts share
    let t2p = 1.0 / (1.0 / 0.05 - 1.0 / 0.08);
    let star = |t2: f64| 1.0 / (1.0 / t2 + 1.0 / t2p);
    let t2a = p.macrovascular.as_ref().unwrap().t2_arterial.0;
    let d = vec![decay(tes, star(p.t2_blood_s.0)), decay(tes, star(0.08)), decay(tes, star(t2a))];
    let (worst, parts) = fit_parts(&echo_images(&out), &d);
    println!("three parts (gradient echo): worst prediction error {worst:e}, parts {parts:?}");
    assert!(worst < 1e-4, "{worst}");
    // the arterial term is present (its bolus is passing at t = 3.6 s)
    assert!(parts[2].abs() > 0.01 * parts[0].abs(), "{parts:?}");
}

/// The identity I_C - I_L = I_B holds at every echo.
#[test]
fn linearity_holds_at_every_echo() {
    for (ge, tes) in [(false, &SE_TES[..2]), (true, &GE_TES[..2])] {
        let run = |row: &str| {
            let out = simulate_with(&multi_te(row, tes, "", ge), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
            echo_images(&out)
        };
        let (c, l, b) = (run("control"), run("label"), run("deltam"));
        for e in 0..tes.len() {
            let worst = residual_of(&c[e], &l[e], &b[e]);
            assert!(worst <= 1.0, "ge {ge} echo {e}: {worst}");
        }
    }
}

/// Under compat each echo has its own image set, bounded before it is built.
#[test]
fn compat_multi_te_images_are_bounded() {
    let sidecars: Vec<Value> = [0.015, 0.030].iter().map(|te| json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": te, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.0], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012
    })).collect();
    let parse_ov = |ov: &str| {
        let ov: Overlay = toml::from_str(ov).unwrap();
        aslscan::protocol::parse_echoes(&sidecars, "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap()
    };
    let p = parse_ov("[compat]\nasldro = true\n[multi_te]\nmax_image_memory_gib = 1e-9\n");
    let e = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap_err();
    assert!(e.contains("max_image_memory_gib") && e.contains("GiB"), "{e}");
    // within the limit, echo 2 is compat's single-echo series at its echo time, noise off
    let p = parse_ov("[compat]\nasldro = true\n");
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let ov: Overlay = toml::from_str("[compat]\nasldro = true\n").unwrap();
    let one = aslscan::protocol::parse_echoes(&sidecars[1..], "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
    let one = simulate_with(&one, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert_eq!(bits(&out.more_echoes[0].mag), bits(&one.mag));
    assert_eq!(bits(&out.more_echoes[0].phase), bits(&one.phase));
    assert_ne!(bits(&out.mag), bits(&one.mag));
}

/// The echo-N dataset: one series per echo from its own sidecar, one aslcontext, a separate M0
/// per echo naming its own series, the ground truth once.
#[test]
fn multi_te_dataset_layout() {
    let mut sidecars: Vec<Value> = Vec::new();
    for te in &SE_TES[..2] {
        let mut s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Separate", "RepetitionTimePreparation": 4.0,
            "EchoTime": te, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.07], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.012
        });
        s["EchoTime"] = json!(te);
        sidecars.push(s);
    }
    let ov: Overlay = toml::from_str("seed = 3\n[acquisition]\noversample = 2\n[m0]\nrepetition_time = 6.0\n").unwrap();
    let p = aslscan::protocol::parse_echoes(&sidecars, "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-multite-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let perf = dir.join("sub-01/perf");
    let read = |f: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(perf.join(f)).unwrap()).unwrap() };
    let mut names: Vec<String> = std::fs::read_dir(&perf).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    names.sort();
    assert_eq!(names, [
        "ground-truth", "sub-01_aslcontext.tsv",
        "sub-01_echo-1_m0scan.json", "sub-01_echo-1_m0scan.nii.gz",
        "sub-01_echo-1_part-mag_asl.json", "sub-01_echo-1_part-mag_asl.nii.gz",
        "sub-01_echo-1_part-phase_asl.json", "sub-01_echo-1_part-phase_asl.nii.gz",
        "sub-01_echo-2_m0scan.json", "sub-01_echo-2_m0scan.nii.gz",
        "sub-01_echo-2_part-mag_asl.json", "sub-01_echo-2_part-mag_asl.nii.gz",
        "sub-01_echo-2_part-phase_asl.json", "sub-01_echo-2_part-phase_asl.nii.gz",
    ]);
    for (e, te) in SE_TES[..2].iter().enumerate() {
        let side = read(&format!("sub-01_echo-{}_part-mag_asl.json", e + 1));
        assert_eq!(side["EchoTime"], json!(te));
        let me = &side["AslscanSimulation"]["MultiEcho"];
        assert_eq!((me["Echo"].clone(), me["EchoTimes"].clone()), (json!(e + 1), json!(&SE_TES[..2])));
        assert_eq!(me["ReceiverSeeds"][0], json!(3u64));
        assert_ne!(me["ReceiverSeeds"][1], json!(3u64));
        let m0 = read(&format!("sub-01_echo-{}_m0scan.json", e + 1));
        assert_eq!(m0["EchoTime"], json!(te));
        assert_eq!(m0["IntendedFor"][0], json!(format!("bids::sub-01/perf/sub-01_echo-{}_part-mag_asl.nii.gz", e + 1)));
    }
    // the two echoes and the two M0 scans differ
    assert_ne!(std::fs::read(perf.join("sub-01_echo-1_m0scan.nii.gz")).unwrap(), std::fs::read(perf.join("sub-01_echo-2_m0scan.nii.gz")).unwrap());
    let gt: Vec<String> = std::fs::read_dir(perf.join("ground-truth")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    assert!(gt.iter().all(|n| !n.contains("echo")), "{gt:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The milestone M acceptance fixtures load from their files (one sidecar per echo) and simulate.
#[test]
fn the_multi_te_fixtures_load_and_run() {
    for (dir, ge) in [("p6_multite", true), ("p6_multite_se", false)] {
        let d = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols")).join(dir);
        let jsons: Vec<std::path::PathBuf> = (1..=3).map(|e| d.join(format!("asl-echo-{e}.json"))).collect();
        let refs: Vec<&Path> = jsons.iter().map(|p| p.as_path()).collect();
        let p = aslscan::protocol::load_with(&refs, &d.join("aslcontext.tsv"), Some(&d.join("overlay.toml")), crop().params.as_ref(), false)
            .unwrap();
        assert!(p.p6_active() && p.echo_times_s.len() == 3, "{dir}");
        assert_eq!(p.multi_te.as_ref().unwrap().refocusing_time_ms.is_none(), ge, "{dir}");
        let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        assert_eq!(out.more_echoes.len(), 2);
        assert!(out.more_echoes.iter().all(|e| e.m0.is_some()), "{dir}: a separate M0 per echo");
    }
}

// ---------------------------------------------------------------- P6 part A: Hadamard

const H_TAU: f64 = 0.25;
const H_PLD: f64 = 1.5;

/// Hadamard on the crop: `order`, `cycles` cycles of equal sub-boli (0.25 s, PLD 1.5 s, so every
/// sub-bolus has arrived at the readout), an
/// m0scan row first with `m0_first`; spin echo, or gradient echo at 60 degrees with `ge`.
fn hadamard_crop(order: usize, cycles: usize, m0_first: bool, extra: &str, ge: bool) -> Protocol {
    let n = order - 1;
    let (mut ld, mut pld, mut ctx) = (Vec::new(), Vec::new(), String::from("volume_type\n"));
    if m0_first {
        ld.push(0.0);
        pld.push(0.0);
        ctx.push_str("m0scan\n");
    }
    for _ in 0..cycles {
        for j in 0..n {
            ld.push(H_TAU);
            pld.push(H_PLD + H_TAU * (n - 1 - j) as f64);
            ctx.push_str("deltam\n");
        }
    }
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
        "BackgroundSuppression": false, "M0Type": if m0_first { "Included" } else { "Absent" },
        "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3,
        "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05],
        "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
    });
    let contrast = if ge { "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n" } else { "" };
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n{contrast}[hadamard]\norder = {order}\n{extra}")).unwrap();
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

/// The single deltam row of sub-bolus `j` of `order`: its own duration and effective delay.
fn single_subbolus(order: usize, j: usize, extra: &str, ge: bool) -> Protocol {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": H_TAU,
        "PostLabelingDelay": H_PLD + H_TAU * (order - 2 - j) as f64,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012,
        "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D",
        "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
    });
    let contrast = if ge { "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n" } else { "" };
    let ov: Overlay = toml::from_str(&format!("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n{contrast}{extra}")).unwrap();
    parse(&s, "volume_type\ndeltam\n", Some(&ov), crop().params.as_ref()).unwrap()
}

/// Decode raw volumes with an arbitrary +-1 matrix (the negative controls' wrong encodings).
fn decode_with(h: &[Vec<i8>], raw: &[Vec<(f64, f64)>], j: usize) -> Vec<(f64, f64)> {
    let scale = 2.0 / raw.len() as f64;
    (0..raw[0].len()).map(|x| {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, r) in raw.iter().enumerate() {
            re += h[i][j] as f64 * r[x].0;
            im += h[i][j] as f64 * r[x].1;
        }
        (scale * re, scale * im)
    }).collect()
}

/// The addendum's per-voxel comparison: `|D - ref| <= 4 * 2^-24 * max_i |S_i| + 1e-6 * max |ref|`,
/// with `max |ref|` above 100 times the first term's largest value (a resolved sub-bolus). Returns
/// the worst ratio of the error to its bound.
fn decode_ratio(d: &[(f64, f64)], reference: &[(f64, f64)], raw: &[Vec<(f64, f64)>]) -> f64 {
    let max_ref = reference.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
    let smax: Vec<f64> = (0..d.len()).map(|x| raw.iter().map(|r| r[x].0.hypot(r[x].1)).fold(0.0f64, f64::max)).collect();
    let first = 4.0 * 2f64.powi(-24);
    let floor = first * smax.iter().cloned().fold(0.0, f64::max);
    assert!(max_ref > 100.0 * floor, "the sub-bolus is not resolved: max ref {max_ref}, f32 floor {floor}");
    d.iter().zip(reference).zip(&smax)
        .map(|((a, b), s)| (a.0 - b.0).hypot(a.1 - b.1) / (first * s + 1e-6 * max_ref))
        .fold(0.0f64, f64::max)
}

fn raw_images(h: &aslscan::series::HadamardSeries, range: std::ops::Range<usize>) -> Vec<Vec<(f64, f64)>> {
    range.map(|r| {
        let n = h.n_raw;
        (0..h.raw_mag.len() / n).map(|x| {
            let (m, p) = (h.raw_mag[x * n + r] as f64, h.raw_phase[x * n + r] as f64);
            (m * p.cos(), m * p.sin())
        }).collect()
    }).collect()
}

fn volume(out: &SeriesOutput, v: usize) -> Vec<(f64, f64)> {
    let all = complex_from(&out.mag, &out.phase);
    let n = out.n_volumes;
    (0..all.len() / n).map(|x| all[x * n + v]).collect()
}

/// Decoded sub-bolus `j` is the single deltam run of sub-bolus `j` (its own duration and
/// effective delay), spin echo and gradient echo in the steady state, to the addendum's
/// tolerance; a flipped sign and a swapped column in the decoding fail the same comparison.
#[test]
fn a_decoded_sub_bolus_is_its_single_deltam_run() {
    let order = 8;
    for ge in [false, true] {
        let p = hadamard_crop(order, 1, false, "", ge);
        let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        let h = out.hadamard.as_ref().unwrap();
        assert_eq!((out.n_volumes, h.n_raw), (order - 1, order));
        let raw = raw_images(h, 0..order);
        let enc = aslscan::hadamard::encoding(order);
        let mut flipped = enc.clone();
        flipped[5][2] = -flipped[5][2];
        let mut swapped = enc.clone();
        for r in swapped.iter_mut() {
            r.swap(1, 4);
        }
        for j in 0..order - 1 {
            let one = simulate_with(&single_subbolus(order, j, "", ge), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
            let reference = volume(&one, 0);
            let worst = decode_ratio(&volume(&out, j), &reference, &raw);
            assert!(worst <= 1.0, "ge {ge} sub-bolus {j}: {worst}");
            for (what, m) in [("flipped", &flipped), ("swapped", &swapped)] {
                if (what == "flipped" && j == 2) || (what == "swapped" && (j == 1 || j == 4)) {
                    let bad = decode_ratio(&decode_with(m, &raw, j), &reference, &raw);
                    assert!(bad > 10.0, "ge {ge} {what} sub-bolus {j}: {bad}");
                }
            }
        }
    }
}

/// With exchange and no transient the tissue of every raw volume is the same, so the decoded
/// tissue-only residual vanishes: the extravascular label (in the tissue compartments) is not
/// counted as leakage.
#[test]
fn no_transient_no_leakage_with_exchange() {
    let p = hadamard_crop(8, 1, false, "[kinetic]\nexchange_time = 0.5\n", false);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let l = &out.hadamard.as_ref().unwrap().leakage.as_ref().unwrap()[0];
    assert!(l.reference_norm > 0.0);
    // numerically zero: identical tissue in every raw volume (the acquisition's arithmetic
    // differs between volumes only at the level of the near-zero background)
    for (j, &(abs, rel)) in l.per_subbolus.iter().enumerate() {
        assert!(rel <= 1e-12, "sub-bolus {j}: {abs} ({rel})");
    }
}

/// Gradient echo with an included M0 before the cycle: the longitudinal state carries over, so
/// the raw volumes' tissue differs and leaks into every decoded sub-bolus. The decoded volume
/// minus the single deltam run is the decoded tissue-only residual.
#[test]
fn a_transient_leaks_and_the_leakage_is_the_decoded_tissue() {
    let order = 8;
    let p = hadamard_crop(order, 1, true, "", true);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let h = out.hadamard.as_ref().unwrap();
    assert!(h.flags.transients);
    let l = &h.leakage.as_ref().unwrap()[0];
    let brain: Vec<bool> = out.ground_truth.dseg.iter().map(|d| *d > 0).collect();
    for j in 0..order - 1 {
        let reference = volume(&simulate_with(&single_subbolus(order, j, "", true), &crop(), T2Mode::Auto, &phase(),
                                              RowOverride::None).unwrap(), 0);
        // output 0 is the m0scan row
        let d = volume(&out, j + 1);
        let resid: f64 = d.iter().zip(&reference).zip(&brain).filter(|(_, b)| **b)
            .map(|((a, r), _)| (a.0 - r.0).powi(2) + (a.1 - r.1).powi(2)).sum::<f64>().sqrt();
        let (abs, rel) = l.per_subbolus[j];
        println!("transient: sub-bolus {j} leakage {abs:.4} ({rel:.2e} of the reference), decoded - single {resid:.4}");
        assert!(abs > 1e-3 * l.reference_norm * 1e-3, "sub-bolus {j}: no leakage");
        assert!((resid - abs).abs() <= 1e-3 * abs + 1e-6 * l.reference_norm, "sub-bolus {j}: {resid} vs {abs}");
    }
}

/// The encoded raw truth is the kinetic sum of the labeled sub-boli, and over all sub-boli it is
/// the whole bolus.
#[test]
fn the_raw_truth_is_the_encoded_sum() {
    let order = 4;
    let p = hadamard_crop(order, 1, false, "", false);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let h = out.hadamard.as_ref().unwrap();
    let n = h.n_raw;
    let nvox = h.raw_delta_m.len() / n;
    let enc = aslscan::hadamard::encoding(order);
    // the decoded truth per sub-bolus, by voxel; raw volume i's truth is the sum of its labeled ones
    for (i, row) in enc.iter().enumerate() {
        let w = aslscan::hadamard::weights(row);
        for x in 0..nvox {
            let want: f64 = (0..order - 1).filter(|&j| w[j] == 1).map(|j| out.ground_truth.delta_m[x * (order - 1) + j] as f64).sum();
            let got = h.raw_delta_m[x * n + i] as f64;
            assert!((got - want).abs() <= 1e-5 * want.abs().max(1e-3), "raw {i} voxel {x}: {got} vs {want}");
        }
    }
    // raw volume 0 labels nothing
    assert!((0..nvox).all(|x| h.raw_delta_m[x * n] == 0.0));
}


/// The Hadamard dataset (2D): the decoded series in the main tree with its Hadamard block and
/// TotalAcquiredPairs = cycles, the raw series and its truth under sourcedata, one row per
/// preparation in the preparation table.
#[test]
fn hadamard_dataset_layout() {
    let order = 4;
    let p = hadamard_crop(order, 2, true, "[physio]\ntissue_cardiac = 0.03\nlabel_cardiac = 0.04\n", false);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-hadamard-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let list = |d: &std::path::Path| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
        v.sort();
        v
    };
    let perf = dir.join("sub-01/perf");
    assert_eq!(list(&perf), ["ground-truth", "sub-01_aslcontext.tsv", "sub-01_part-mag_asl.json", "sub-01_part-mag_asl.nii.gz",
                             "sub-01_part-phase_asl.json", "sub-01_part-phase_asl.nii.gz"]);
    // the per-volume tables are the raw volumes', so they are in sourcedata only
    assert!(!list(&perf.join("ground-truth")).iter().any(|f| f.contains("physio") || f.contains("preparations")));
    assert_eq!(std::fs::read_to_string(perf.join("sub-01_aslcontext.tsv")).unwrap(),
               format!("volume_type\nm0scan\n{}", "deltam\n".repeat(2 * (order - 1))));
    let side: Value = serde_json::from_str(&std::fs::read_to_string(perf.join("sub-01_part-mag_asl.json")).unwrap()).unwrap();
    assert_eq!(side["TotalAcquiredPairs"], json!(2));
    let hb = &side["AslscanSimulation"]["Hadamard"];
    assert_eq!((hb["Order"].clone(), hb["Counts"].clone()), (json!(4), json!({ "Preparations": 9, "RawVolumes": 9, "Decoded": 6 })));
    assert_eq!(hb["Outputs"][0], json!({ "RawVolume": 0 }));
    assert_eq!(hb["Outputs"][4], json!({ "Cycle": 2, "SubBolus": 1 }));
    // H4 row 1 is (-1, 1, -1) without the all-ones column: sub-boli 1 and 3 labeled
    assert_eq!(hb["RawVolumes"][2]["LabeledSubBoli"], json!([1, 3]));
    assert_eq!(hb["NonExact"]["Physiology"], json!(true));
    assert!(hb["TissueLeakage"].as_array().unwrap().len() == 2 && hb["TotalAcquiredPairsConvention"].is_string());
    let gt: Value = serde_json::from_str(&std::fs::read_to_string(perf.join("ground-truth/sub-01_desc-deltam_gt.json")).unwrap()).unwrap();
    assert!(gt["Description"].as_str().unwrap().contains("ideal sub-bolus truth"));
    // sourcedata: the raw series, one row per raw volume, the raw truth, the preparations
    let src = dir.join("sourcedata/sub-01/perf");
    assert_eq!(list(&src), ["ground-truth", "sub-01_part-mag_asl.json", "sub-01_part-mag_asl.nii.gz", "sub-01_part-phase_asl.json",
                            "sub-01_part-phase_asl.nii.gz", "sub-01_rawvolumes.tsv"]);
    let raw = std::fs::read_to_string(src.join("sub-01_rawvolumes.tsv")).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 10);
    assert_eq!(lines[1], "0\tm0scan\tn/a\tn/a\t\t0\t1");
    assert_eq!(lines[2], "1\tencoded\t1\t0\t\t1\t1");
    assert_eq!(lines[4], "3\tencoded\t1\t2\t2,3\t3\t1");
    let srcgt = list(&src.join("ground-truth"));
    for f in ["sub-01_desc-deltam_gt.nii.gz", "sub-01_desc-physio_gt.tsv", "sub-01_desc-preparations_gt.tsv"] {
        assert!(srcgt.contains(&f.to_string()), "{f} in {srcgt:?}");
    }
    let prep = std::fs::read_to_string(src.join("ground-truth/sub-01_desc-preparations_gt.tsv")).unwrap();
    let rows: Vec<Vec<&str>> = prep.lines().skip(1).map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 9);
    let h = out.hadamard.as_ref().unwrap();
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r[0], i.to_string());
        assert_eq!(r[1], h.prep_factors[i].raw.to_string());
        // the label factor recorded is the one applied, and it varies (physiology)
        assert_eq!(r[8].parse::<f64>().unwrap(), h.prep_factors[i].label);
        assert_eq!(r[9], "per slice (desc-physio_gt.tsv)");
    }
    let nii = |f: &std::path::Path| -> usize {
        nifti::ReaderOptions::new().read_file(f).unwrap().header().dim[4] as usize
    };
    assert_eq!(nii(&src.join("ground-truth/sub-01_desc-deltam_gt.nii.gz")), 9);
    assert_eq!(nii(&perf.join("ground-truth/sub-01_desc-deltam_gt.nii.gz")), 7);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The milestone H acceptance fixtures on the crop load from their files and simulate.
#[test]
fn the_hadamard_fixtures_load_and_run() {
    let base = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols"));
    for (dir, echoes) in [("p6_hadamard", 1usize), ("p6_hadamard_multite", 3)] {
        let d = base.join(dir);
        let jsons: Vec<std::path::PathBuf> = if echoes == 1 {
            vec![d.join("asl.json")]
        } else {
            (1..=echoes).map(|e| d.join(format!("asl-echo-{e}.json"))).collect()
        };
        let refs: Vec<&Path> = jsons.iter().map(|p| p.as_path()).collect();
        let p = aslscan::protocol::load_with(&refs, &d.join("aslcontext.tsv"), Some(&d.join("overlay.toml")), crop().params.as_ref(), false)
            .unwrap();
        let h = p.hadamard.as_ref().unwrap();
        assert_eq!((h.order, h.cycles.len(), p.echo_times_s.len()), (8, 2, echoes), "{dir}");
        let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        assert_eq!((out.n_volumes, out.hadamard.as_ref().unwrap().n_raw, out.more_echoes.len()), (15, 17, echoes - 1));
    }
    // the 3 T variants parse (their phantoms are local)
    for dir in ["p6_hadamard_3t", "p6_hadamard_grase"] {
        let d = base.join(dir);
        let p = aslscan::protocol::load_with(&[d.join("asl.json").as_path()], &d.join("aslcontext.tsv"), Some(&d.join("overlay.toml")),
                                             None, false).unwrap();
        assert!(p.hadamard.is_some(), "{dir}");
    }
}

// ---------------------------------------------------------------- P6 part B: Look-Locker

/// A Look-Locker PCASL series on the crop: `rows` (one cycle per run of a volume type), `m`
/// readouts 0.3 s apart from PLD 0.6 s, gradient echo at `flip` degrees, repetition time `tr`.
fn look_locker(rows: &[&str], m: usize, flip: f64, tr: f64, extra: &str) -> Protocol {
    let mut pld = Vec::new();
    let mut ctx = String::from("volume_type\n");
    for kind in rows {
        for n in 0..m {
            pld.push(0.6 + 0.3 * n as f64);
            ctx.push_str(kind);
            ctx.push('\n');
        }
    }
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.0, "PostLabelingDelay": pld,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": tr, "LookLocker": true,
        "EchoTime": 0.012, "FlipAngle": flip, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
    });
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n[signal]\nacq_contrast = \"ge\"\n{extra}")).unwrap();
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

/// At a vanishing flip the readouts deplete nothing: the blood read over sin(a) is the undepleted
/// delta_m at each slice's excitation, and the tissue over sin(a) is the fully recovered M0
/// (no saturation; an independent reference, not the timeline).
#[test]
fn a_vanishing_flip_reads_the_undepleted_label() {
    let a: f64 = 1e-3;
    let p = look_locker(&["label"], 8, a, 4.5, "");
    let out = simulate_with(&p, &crop(), T2Mode::Class, &phase(), RowOverride::None).unwrap();
    let ll = out.look_locker.as_ref().unwrap();
    assert!(!ll.legacy_dispatch);
    let read = ll.delta_m_read.as_ref().unwrap();
    let truth = &out.ground_truth.delta_m;
    let peak = truth.iter().cloned().fold(0.0f32, f32::max) as f64;
    assert!(peak > 0.0, "a nonzero reference");
    let sin = a.to_radians().sin();
    let worst = read.iter().zip(truth).map(|(r, t)| (*r as f64 / sin - *t as f64).abs()).fold(0.0f64, f64::max) / peak;
    println!("vanishing flip: read / sin(a) vs undepleted, worst {worst:e} of the peak");
    assert!(worst < 1e-5, "{worst}");
    // and at 40 degrees the later readouts read visibly less than the undepleted label
    let p40 = look_locker(&["label"], 8, 40.0, 4.5, "");
    let o40 = simulate_with(&p40, &crop(), T2Mode::Class, &phase(), RowOverride::None).unwrap();
    let r40 = o40.look_locker.as_ref().unwrap().delta_m_read.as_ref().unwrap();
    let n = o40.n_volumes;
    let sum = |v: &[f32], k: usize| (0..v.len() / n).map(|x| v[x * n + k] as f64).sum::<f64>();
    let s40 = 40f64.to_radians().sin();
    assert!((sum(r40, 0) / s40 - sum(&o40.ground_truth.delta_m, 0)).abs() < 1e-4 * sum(&o40.ground_truth.delta_m, 0));
    assert!(sum(r40, 7) / s40 < 0.9 * sum(&o40.ground_truth.delta_m, 7));
    // the tissue: with nothing saturating it the steady state is M0 itself; the control cycle's
    // compartments over sin(a), integrated, against the M0 truth integrated (box averages keep the
    // integral; a simulation voxel is a quarter of an acquisition voxel)
    let pc = look_locker(&["control"], 8, a, 4.5, "");
    let (comps, oc) = aslscan::series::simulate_compartments(&pc, &crop(), T2Mode::Class, &phase(), RowOverride::None).unwrap();
    let (k, nv) = (oc.labels.len(), oc.n_volumes);
    let m0_sum: f64 = oc.ground_truth.m0.iter().map(|x| *x as f64).sum();
    for v in 0..nv {
        let tissue: f64 = comps[..k].iter().map(|c| c.iter().skip(v).step_by(nv).map(|x| *x as f64).sum::<f64>()).sum::<f64>() / sin / 4.0;
        assert!((tissue - m0_sum).abs() <= 1e-4 * m0_sum, "readout {v}: {tissue} vs {m0_sum}");
    }
}

/// The identity I_C - I_L = I_B per readout holds when the control and label cycles share their
/// history (matched preparation), fails for a flipped label sign, and fails when the label
/// cycle's history differs (another repetition time): the depleted tissue does not cancel.
#[test]
fn look_locker_linearity_per_readout() {
    let m = 6;
    let run = |rows: &[&str], tr: f64, ov: RowOverride| {
        let out = simulate_with(&look_locker(rows, m, 30.0, tr, ""), &crop(), T2Mode::Auto, &phase(), ov).unwrap();
        let all = complex_from(&out.mag, &out.phase);
        let n = out.n_volumes;
        (0..n).map(|v| (0..all.len() / n).map(|x| all[x * n + v]).collect::<Vec<_>>()).collect::<Vec<_>>()
    };
    let (c, l, b) = (run(&["control"], 4.5, RowOverride::None), run(&["label"], 4.5, RowOverride::None), run(&["deltam"], 4.5, RowOverride::None));
    for n in 0..m {
        let worst = residual_of(&c[n], &l[n], &b[n]);
        assert!(worst <= 1.0, "readout {n}: {worst}");
    }
    let flipped = run(&["label"], 4.5, RowOverride::FlipLabelSign);
    assert!(residual_of(&c[m - 1], &flipped[m - 1], &b[m - 1]) > 1e2);
    let other = run(&["label"], 4.0, RowOverride::None);
    assert!(residual_of(&c[m - 1], &other[m - 1], &b[m - 1]) > 1e2, "mismatched history must not cancel");
}

/// One readout per cycle at one flip is P5's gradient-echo series bit for bit: p5_ge (two delays,
/// suppression, a separate M0, 60 degrees: the state carried row to row) with LookLocker: true.
#[test]
fn one_readout_look_locker_is_p5() {
    let d = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/p5_ge/");
    let s: Value = serde_json::from_str(&std::fs::read_to_string(format!("{d}asl.json")).unwrap()).unwrap();
    let ctx = std::fs::read_to_string(format!("{d}aslcontext.tsv")).unwrap();
    let ov: Overlay = toml::from_str(&std::fs::read_to_string(format!("{d}overlay.toml")).unwrap()).unwrap();
    let base = parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap();
    let mut sl = s.clone();
    sl["LookLocker"] = json!(true);
    let ll = parse(&sl, &ctx, Some(&ov), crop().params.as_ref()).unwrap();
    assert!(ll.p6_active() && ll.look_locker.as_ref().unwrap().cycles.iter().all(|c| c.rows.len() == 1));
    let (a, b) = (simulate_with(&base, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap(),
                  simulate_with(&ll, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap());
    assert!(b.look_locker.as_ref().unwrap().legacy_dispatch);
    assert_eq!(bits(&a.mag), bits(&b.mag));
    assert_eq!(bits(&a.phase), bits(&b.phase));
    assert_eq!(bits(&a.m0.as_ref().unwrap().0), bits(&b.m0.as_ref().unwrap().0));
    assert_eq!(bits(&a.ground_truth.delta_m), bits(&b.ground_truth.delta_m));
}

/// The milestone L fixture loads and runs; its separate M0 takes the series' flip.
#[test]
fn the_look_locker_fixture_loads_and_runs() {
    let d = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/protocols/p6_ll"));
    let p = aslscan::protocol::load_with(&[d.join("asl.json").as_path()], &d.join("aslcontext.tsv"), Some(&d.join("overlay.toml")),
                                         crop().params.as_ref(), false).unwrap();
    let l = p.look_locker.as_ref().unwrap();
    assert_eq!((l.cycles.len(), l.readouts_per_cycle, l.m0_flip_deg.map(|f| f.0)), (2, Some(12), Some(35.0)));
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert_eq!(out.look_locker.as_ref().unwrap().lines.len(), 2 * 12 * 2);
}

/// Review fixes, end to end: Hadamard motion covers every raw volume; a constant FlipAngle array
/// takes the generalized Look-Locker timeline (the legacy dispatch is a scalar flip); the
/// Look-Locker table is per phantom label in voxel mode; the preparation table's header; each
/// echo's acquisition record carries its own TE.
#[test]
fn review_fixes_end_to_end() {
    // every raw volume moves (default volumes: all 8 raw volumes of one H4 x 2 cycle series)
    let p = hadamard_crop(4, 2, false, "[motion]\nmode = \"random\"\ntrans_mm = [1.0, 1.0, 0.0]\nrot_deg = [0.0, 0.0, 2.0]\n", false);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert_eq!(out.poses.len(), 8);
    assert!(out.poses.iter().all(|q| q.trans_mm != [0.0; 3]), "{:?}", out.poses);

    // one readout per cycle with a constant FlipAngle array: the generalized timeline
    let mut arr = look_locker(&["control", "label"], 1, 35.0, 4.5, "");
    {
        let l = arr.look_locker.as_mut().unwrap();
        l.flip_array = true;
    }
    let o = simulate_with(&arr, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert!(!o.look_locker.as_ref().unwrap().legacy_dispatch);
    let sc = simulate_with(&look_locker(&["control", "label"], 1, 35.0, 4.5, ""), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert!(sc.look_locker.as_ref().unwrap().legacy_dispatch);

    // voxel mode: the table has one mean per phantom label, and they differ
    let v = simulate_with(&look_locker(&["control"], 4, 35.0, 4.5, ""), &crop(), T2Mode::Voxel, &phase(), RowOverride::None).unwrap();
    let lines = &v.look_locker.as_ref().unwrap().lines;
    assert!(lines.iter().all(|l| l.tissue_mz.len() == v.labels.len()));
    assert!((lines[0].tissue_mz[0] - lines[0].tissue_mz[1]).abs() > 1e-6, "{:?}", lines[0].tissue_mz);

    // the preparation table's header, by name
    let p = hadamard_crop(4, 1, false, "", false);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-review-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    let prep = std::fs::read_to_string(dir.join("sourcedata/sub-01/perf/ground-truth/sub-01_desc-preparations_gt.tsv")).unwrap();
    assert_eq!(prep.lines().next().unwrap().split('\t').collect::<Vec<_>>(),
               ["preparation", "raw_volume", "shot", "encoding_row", "labeled_subboli", "start", "labeling_window_start",
                "labeling_window_end", "label_factor", "tissue_factor", "suppression_factor", "shot_gain"]);
    let raw = std::fs::read_to_string(dir.join("sourcedata/sub-01/perf/sub-01_rawvolumes.tsv")).unwrap();
    assert_eq!(raw.lines().next().unwrap().split('\t').count(), 7);
    let _ = std::fs::remove_dir_all(&dir);

    // each echo's sidecar: its EchoTime and its acquisition TE agree
    let p = multi_te("control,label", &SE_TES[..2], "", false);
    let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
    for (e, te) in SE_TES[..2].iter().enumerate() {
        let side: Value = serde_json::from_str(&std::fs::read_to_string(
            dir.join(format!("sub-01/perf/sub-01_echo-{}_part-mag_asl.json", e + 1))).unwrap()).unwrap();
        assert!((side["AslscanSimulation"]["Acquisition"]["TEchoMs"].as_f64().unwrap() - te * 1000.0).abs() < 1e-9);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Follow-up review: the compat image bound is checked before anything is built (a phantom with a
/// fieldmap, which the build refuses under compat, still gets the bound's error); the legacy
/// Look-Locker sidecar says which truths it writes, and writes none of the others; a constant
/// FlipAngle array with an included M0 between cycles takes the generalized timeline numerically.
#[test]
fn follow_up_review_fixes() {
    let sidecars: Vec<Value> = [0.015, 0.030].iter().map(|te| json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": te, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.0], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012
    })).collect();
    let ov: Overlay = toml::from_str("[compat]\nasldro = true\n[multi_te]\nmax_image_memory_gib = 1e-9\n").unwrap();
    let p = aslscan::protocol::parse_echoes(&sidecars, "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
    let mut ph = crop();
    ph.fieldmap = Some(vec![0.0; ph.dseg.len()]);
    let e = simulate_with(&p, &ph, T2Mode::Auto, &phase(), RowOverride::None).unwrap_err();
    assert!(e.contains("max_image_memory_gib") && !e.contains("fieldmap"), "{e}");

    // the legacy Look-Locker sidecar and files, and the generalized one's
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-followup-{}", std::process::id()));
    for (m, legacy) in [(1usize, true), (4, false)] {
        let p = look_locker(&["control", "label"], m, 35.0, 4.5, "");
        let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
        let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
        let gt = side["AslscanSimulation"]["LookLocker"]["GroundTruth"].as_str().unwrap().to_string();
        let files: Vec<String> = std::fs::read_dir(dir.join("sub-01/perf/ground-truth")).unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
        let extra = files.iter().any(|f| f.contains("deltamRead") || f.contains("lookLocker"));
        assert_eq!((gt.contains("legacy dispatch writes no"), extra), (legacy, !legacy), "m {m}: {gt} {files:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);

    // one readout per cycle, an m0scan between: the scalar takes P5's M0 convention, the array the
    // generalized one, so the images differ
    let run = |flip: Value| {
        let s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.0, "PostLabelingDelay": [0.6, 0.0, 0.6],
            "BackgroundSuppression": false, "M0Type": "Included", "RepetitionTimePreparation": 4.5, "LookLocker": true,
            "EchoTime": 0.012, "FlipAngle": flip, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
        });
        let ov: Overlay = toml::from_str("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n[signal]\nacq_contrast = \"ge\"\n").unwrap();
        let p = parse(&s, "volume_type\ncontrol\nm0scan\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap();
        simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap()
    };
    let (sc, ar) = (run(json!(35)), run(json!([35, 35, 35])));
    assert!(sc.look_locker.as_ref().unwrap().legacy_dispatch && !ar.look_locker.as_ref().unwrap().legacy_dispatch);
    assert_ne!(bits(&sc.mag), bits(&ar.mag));
}

/// P7 part A: a Look-Locker series with exchange and the arterial compartment writes the read by
/// part, and its sidecar lists the parts in force in place of P6's refusals; P6's Look-Locker
/// series keeps its sidecar block.
#[test]
fn look_locker_with_p4_parts_writes_the_parts() {
    let parts = "[kinetic]\nexchange_time = 0.6\n[macrovascular]\n\
                 arterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
                 arterial_transit_time = { grey_matter = 1.2, white_matter = 1.8, csf = 0.0 }\n";
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-p7a-{}", std::process::id()));
    for (extra, on) in [(parts, true), ("", false)] {
        let p = look_locker(&["control", "label"], 4, 35.0, 4.5, extra);
        let out = simulate_with(&p, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &p, &out).unwrap();
        let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_part-mag_asl.json")).unwrap()).unwrap();
        let ll = &side["AslscanSimulation"]["LookLocker"];
        let files: Vec<String> = std::fs::read_dir(dir.join("sub-01/perf/ground-truth")).unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
        for part in ["deltamReadIV", "deltamReadEV", "arterialRead"] {
            assert_eq!(files.iter().any(|f| f.contains(&format!("desc-{part}_gt.nii.gz"))), on, "{part}: {files:?}");
        }
        assert_eq!(ll.get("Refused").is_some(), !on, "{ll}");
        assert_eq!(ll.get("P4Parts").is_some(), on, "{ll}");
        if on {
            let listed: Vec<&str> = ll["P4Parts"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
            assert_eq!(listed, ["exchange (P4 part A)", "the arterial compartment (P4 part B)"]);
            assert!(ll["FreshArterial"].as_str().unwrap().contains("QUASAR"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// P7 part B: a Look-Locker series read at three echo times. With noise off each echo is the
/// single-echo Look-Locker series at its echo time, bit for bit; the truths are echo-independent
/// and written once; compat stays refused.
#[test]
fn look_locker_reads_every_echo() {
    let m = 4;
    let mut pld = Vec::new();
    let mut ctx = String::from("volume_type\n");
    for kind in ["control", "label"] {
        for n in 0..m {
            pld.push(0.6 + 0.3 * n as f64);
            ctx.push_str(kind);
            ctx.push('\n');
        }
    }
    let base = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.0, "PostLabelingDelay": pld,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.5, "LookLocker": true,
        "EchoTime": 0.012, "FlipAngle": 35, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
    });
    let tes = [0.012, 0.024, 0.036];
    let sidecars: Vec<Value> = tes.iter().map(|te| { let mut s = base.clone(); s["EchoTime"] = json!(te); s }).collect();
    let ov: Overlay = toml::from_str("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\nnoise_variance = 0.0\n\
                                      [signal]\nacq_contrast = \"ge\"\n[kinetic]\nexchange_time = 0.5\n").unwrap();
    let parse_e = |s: &[Value]| aslscan::protocol::parse_echoes(s, &ctx, Some(&ov), crop().params.as_ref()).unwrap();
    let multi = parse_e(&sidecars);
    let out = simulate_with(&multi, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert_eq!(out.more_echoes.len(), 2);
    for (e, s) in sidecars.iter().enumerate() {
        let one = simulate_with(&parse_e(std::slice::from_ref(s)), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        let (mag, ph) = if e == 0 { (&out.mag, &out.phase) } else { (&out.more_echoes[e - 1].mag, &out.more_echoes[e - 1].phase) };
        assert_eq!(bits(mag), bits(&one.mag), "echo {}", e + 1);
        assert_eq!(bits(ph), bits(&one.phase), "echo {}", e + 1);
    }
    // the echoes differ (the test is not comparing one image three times)
    assert_ne!(bits(&out.mag), bits(&out.more_echoes[1].mag));
    // one set of truths, no echo entity on them
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-p7b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &multi, &out).unwrap();
    let gt: Vec<String> = std::fs::read_dir(dir.join("sub-01/perf/ground-truth")).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    assert_eq!(gt.iter().filter(|f| f.contains("desc-deltamRead_gt.nii.gz")).count(), 1, "{gt:?}");
    assert_eq!(gt.iter().filter(|f| f.contains("desc-deltamReadIV_gt.nii.gz")).count(), 1, "{gt:?}");
    assert!(gt.iter().all(|f| !f.contains("echo-")), "{gt:?}");
    let _ = std::fs::remove_dir_all(&dir);
    // compat stays refused with Look-Locker, with one echo or several
    let cov: Overlay = toml::from_str("[signal]\nacq_contrast = \"ge\"\n[compat]\nasldro = true\n").unwrap();
    let mut c = sidecars.clone();
    for s in c.iter_mut() {
        s["SliceTiming"] = json!([0.0, 0.0]);
    }
    let err = aslscan::protocol::parse_echoes(&c, &ctx, Some(&cov), crop().params.as_ref()).unwrap_err();
    assert!(err.contains("compat") || err.contains("asldro"), "{err}");
}

/// A Look-Locker Hadamard-4 PCASL sidecar on the crop: three 0.3 s sub-boli read at PLD_n = 1.0,
/// 1.3, 1.6 s (35 degrees), two encoding cycles, the context the decoded volumes readout-major.
fn ll_hadamard_side(te: f64) -> (Value, String) {
    let (mut ld, mut pld, mut ctx) = (Vec::new(), Vec::new(), String::from("volume_type\n"));
    for _ in 0..2 {
        for p_n in [1.0, 1.3, 1.6] {
            for j in 0..3 {
                ld.push(0.3);
                pld.push(p_n + 0.3 * (2 - j) as f64);
                ctx.push_str("deltam\n");
            }
        }
    }
    (json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "LookLocker": true,
        "EchoTime": te, "FlipAngle": 35, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
    }), ctx)
}

/// P7 part B: Look-Locker x Hadamard x multi-TE. With noise off each echo's decoded series is the
/// single-echo series at its echo time, bit for bit; the dataset lists 18 decoded volumes,
/// TotalAcquiredPairs is the number of encoding cycles, and the raw-volume table has a readout
/// column.
#[test]
fn look_locker_hadamard_reads_every_echo() {
    let ov: Overlay = toml::from_str("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\nnoise_variance = 0.0\n\
        [signal]\nacq_contrast = \"ge\"\n[hadamard]\norder = 4\n[look_locker]\nreadouts_per_cycle = 3\n").unwrap();
    let tes = [0.012, 0.024];
    let ctx = ll_hadamard_side(0.012).1;
    let sides: Vec<Value> = tes.iter().map(|&te| ll_hadamard_side(te).0).collect();
    let parse_e = |s: &[Value]| aslscan::protocol::parse_echoes(s, &ctx, Some(&ov), crop().params.as_ref()).unwrap();
    let multi = parse_e(&sides);
    let out = simulate_with(&multi, &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
    assert_eq!(out.n_volumes, 18);
    for (e, s) in sides.iter().enumerate() {
        let one = simulate_with(&parse_e(std::slice::from_ref(s)), &crop(), T2Mode::Auto, &phase(), RowOverride::None).unwrap();
        let (mag, ph) = if e == 0 { (&out.mag, &out.phase) } else { (&out.more_echoes[e - 1].mag, &out.more_echoes[e - 1].phase) };
        assert_eq!(bits(mag), bits(&one.mag), "echo {}", e + 1);
        assert_eq!(bits(ph), bits(&one.phase), "echo {}", e + 1);
    }
    let dir = std::env::temp_dir().join(format!("aslscan-e2e-p7llh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    aslscan::bids::write_dataset(&dir, &aslscan::bids::Names::new("01", None), &multi, &out).unwrap();
    let side: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sub-01/perf/sub-01_echo-1_part-mag_asl.json")).unwrap()).unwrap();
    assert_eq!(side["TotalAcquiredPairs"], json!(2));
    let had = &side["AslscanSimulation"]["Hadamard"];
    assert_eq!(had["Readouts"], json!(3));
    assert_eq!(had["Outputs"][3], json!({ "Cycle": 1, "SubBolus": 1, "Readout": 2 }));
    let raw = std::fs::read_dir(dir.join("sourcedata")).unwrap().flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
        .flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
        .map(|f| f.unwrap().path()).find(|p| p.to_string_lossy().ends_with("_rawvolumes.tsv")).expect("the raw-volume table");
    let tsv = std::fs::read_to_string(raw).unwrap();
    assert!(tsv.lines().next().unwrap().ends_with("\treadout"), "{tsv}");
    assert_eq!(tsv.lines().count(), 1 + 24);
    let _ = std::fs::remove_dir_all(&dir);
}
