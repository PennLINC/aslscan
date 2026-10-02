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
