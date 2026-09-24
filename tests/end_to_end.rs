//! End-to-end properties on the checked-in crop (features `io` + `test-hooks`): the
//! blood-compartment linearity identity with its negative controls, and the noise-variance
//! ratio of control minus label.
#![cfg(all(feature = "io", feature = "test-hooks"))]

use std::path::Path;

use aslscan::phantom::{self, Phantom, T2Mode};
use aslscan::protocol::{parse, Overlay, Protocol};
use aslscan::series::{complex_from, simulate_with, RowOverride};
use mrsim_acq::phase::PhaseModel;
use serde_json::json;

fn crop() -> Phantom {
    phantom::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap()
}

/// PCASL on the crop at 2 mm in-plane, 3 mm slices (12 x 12 x 2), oversample 2, no noise, no
/// GRAPPA, no spikes: the settings the linearity property requires.
fn protocol(rows: &str, noise: f64) -> Protocol {
    let s = json!({
        "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
        "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
        "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
        "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.012
    });
    let ov: Overlay = toml::from_str(&format!(
        "seed = 11\n[acquisition]\noversample = 2\nnoise_variance = {noise}\nsignal_scale = 100.0\npartial_fourier = 0.75\n"
    )).unwrap();
    let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
    parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
}

fn phase() -> PhaseModel {
    PhaseModel { global: 0.0, background: Default::default(), prep: None }
}

/// Complex image of the single volume of a one-row series.
fn image(rows: &str, ov: RowOverride) -> Vec<(f64, f64)> {
    let out = simulate_with(&protocol(rows, 0.0), &crop(), T2Mode::Auto, &phase(), ov).unwrap();
    assert_eq!(out.n_volumes, 1);
    complex_from(&out.mag, &out.phase)
}

/// max |I_C - I_L - I_B| against the spec's tolerance; returns (max violation ratio, n voxels).
fn linearity_residual(ov_c: RowOverride, ov_l: RowOverride, ov_b: RowOverride) -> f64 {
    let c = image("control", ov_c);
    let l = image("label", ov_l);
    let b = image("deltam", ov_b);
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
