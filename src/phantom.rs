//! The BIDS-derivatives phantom: one NIfTI + `Units` sidecar per map, `dseg.json` label names,
//! `phantom.json` kinetic constants (spec: phantom contract). Loaded values stay in the units
//! the files carry (seconds); the conversion to the acquisition stage's milliseconds happens in
//! [`Phantom::t2prime_ms`] and [`Phantom::relaxation`], nowhere else.
//!
//! Two representations of transverse relaxation are supported, chosen from the phantom itself
//! (spec: P0 change 1). `class` decomposes the tissue into one compartment per segmentation
//! label, each with a uniform T2 and T2', which keeps the acquisition stage on its fast per-line
//! path and approximates nothing when the maps are constant within each label. `voxel` keeps one
//! tissue compartment with per-voxel maps. `auto` runs the constancy test and picks.

use std::path::Path;

use mrsim_acq::grid::Grid;
use serde_json::Value;

use crate::protocol::PhantomParams;

/// Map name -> the unit string its sidecar must carry.
const MAPS: [(&str, &str); 7] = [
    ("perfusion", "ml/100g/min"),
    ("att", "s"),
    ("T1map", "s"),
    ("T2map", "s"),
    ("T2starmap", "s"),
    ("M0map", "arbitrary"),
    ("dseg", "label indices"),
];

#[derive(Debug, Clone)]
pub struct Phantom {
    /// Dimensions and affine of every map.
    pub grid: Grid,
    /// ml/100g/min
    pub perfusion: Vec<f32>,
    /// s
    pub att: Vec<f32>,
    /// s
    pub t1: Vec<f32>,
    /// s
    pub t2: Vec<f32>,
    /// s
    pub t2star: Vec<f32>,
    pub m0: Vec<f32>,
    pub dseg: Vec<i32>,
    /// Hz, if `fieldmap.nii.gz` exists.
    pub fieldmap: Option<Vec<f32>>,
    /// Foreground labels present in `dseg`, ascending, with their names (unnamed labels get
    /// `label-N`).
    pub labels: Vec<(i32, String)>,
    pub params: Option<PhantomParams>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T2Mode {
    Auto,
    Class,
    Voxel,
}

impl T2Mode {
    pub fn as_str(&self) -> &'static str {
        match self {
            T2Mode::Auto => "auto",
            T2Mode::Class => "class",
            T2Mode::Voxel => "voxel",
        }
    }
}

/// Transverse relaxation for the acquisition stage, in MILLISECONDS.
#[derive(Debug, Clone, PartialEq)]
pub enum Relaxation {
    /// One `(T2, T2')` per foreground label, in [`Phantom::labels`] order.
    Class { t2_ms: Vec<f32>, t2p_ms: Vec<f32> },
    /// Per-voxel maps on the PHANTOM grid; `resample` turns them into simulation-grid maps.
    Voxel { t2_ms: Vec<f32>, t2p_ms: Vec<f32> },
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn same_affine(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> bool {
    a.iter().zip(b).all(|(ra, rb)| ra.iter().zip(rb).all(|(x, y)| (x - y).abs() <= 1e-6))
}

/// Load a phantom directory.
pub fn load(dir: &Path) -> Result<Phantom, String> {
    let mut vols: Vec<Vec<f32>> = Vec::with_capacity(MAPS.len());
    let mut grid: Option<Grid> = None;
    for (name, unit) in MAPS {
        let nii = dir.join(format!("{name}.nii.gz"));
        let json = dir.join(format!("{name}.json"));
        if !nii.exists() {
            return Err(format!("phantom: missing map {}", nii.display()));
        }
        let side = read_json(&json)?;
        match side.get("Units").and_then(Value::as_str) {
            Some(u) if u == unit => {}
            Some(u) => return Err(format!("phantom: {name}.json Units is {u:?}, expected {unit:?}")),
            None => return Err(format!("phantom: {name}.json has no Units")),
        }
        let (data, g) = mrsim_acq::io::load_volume(&nii).map_err(|e| format!("phantom: {name}: {e}"))?;
        match &grid {
            None => grid = Some(g),
            Some(g0) => {
                if g.dims != g0.dims {
                    return Err(format!("phantom: {name} is {:?} but the other maps are {:?}", g.dims, g0.dims));
                }
                if !same_affine(&g.voxel_to_world, &g0.voxel_to_world) {
                    return Err(format!("phantom: {name} has a different affine from the other maps"));
                }
            }
        }
        for (i, v) in data.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("phantom: {name} voxel {i} is not finite"));
            }
        }
        vols.push(data);
    }
    let grid = grid.unwrap();
    let mut it = vols.into_iter();
    let (perfusion, att, t1, t2, t2star, m0) =
        (it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
    let dseg: Vec<i32> = it.next().unwrap().iter().map(|v| v.round() as i32).collect();

    // Label names from dseg.json, if present.
    let dseg_side = read_json(&dir.join("dseg.json"))?;
    let names = dseg_side.get("LabelMap").and_then(Value::as_object);
    let mut present: Vec<i32> = dseg.iter().copied().filter(|l| *l > 0).collect();
    present.sort_unstable();
    present.dedup();
    if dseg.iter().any(|l| *l < 0) {
        return Err("phantom: dseg holds a negative label".to_string());
    }
    let labels: Vec<(i32, String)> = present
        .iter()
        .map(|l| {
            let name = names
                .and_then(|m| m.get(&l.to_string()))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("label-{l}"));
            (*l, name)
        })
        .collect();
    if labels.is_empty() {
        return Err("phantom: dseg has no foreground labels".to_string());
    }

    // Foreground positivity (spec: T2' derivation table) and background zero M0.
    for (i, &l) in dseg.iter().enumerate() {
        if l > 0 {
            for (name, v) in [("T1map", t1[i]), ("T2map", t2[i]), ("T2starmap", t2star[i]), ("M0map", m0[i])] {
                if v <= 0.0 {
                    return Err(format!("phantom: {name} is {v} at foreground voxel {i} (label {l}); must be positive"));
                }
            }
            if perfusion[i] < 0.0 || att[i] < 0.0 {
                return Err(format!("phantom: negative perfusion or att at voxel {i} (label {l})"));
            }
        } else if m0[i] != 0.0 {
            return Err(format!("phantom: M0map is {} at background voxel {i}; background must have zero magnetization", m0[i]));
        }
    }

    let fm_path = dir.join("fieldmap.nii.gz");
    let fieldmap = if fm_path.exists() {
        let side = read_json(&dir.join("fieldmap.json"))?;
        match side.get("Units").and_then(Value::as_str) {
            Some("Hz") => {}
            other => return Err(format!("phantom: fieldmap.json Units is {other:?}, expected \"Hz\"")),
        }
        let (data, g) = mrsim_acq::io::load_volume(&fm_path).map_err(|e| format!("phantom: fieldmap: {e}"))?;
        if g.dims != grid.dims || !same_affine(&g.voxel_to_world, &grid.voxel_to_world) {
            return Err("phantom: fieldmap is not on the phantom grid".to_string());
        }
        Some(data)
    } else {
        None
    };

    let pj = dir.join("phantom.json");
    let params = if pj.exists() {
        let v = read_json(&pj)?;
        let f = |k: &str| v.get(k).and_then(Value::as_f64);
        Some(PhantomParams {
            lambda: f("LambdaBloodBrain"),
            t1b: f("T1ArterialBlood"),
            field_strength: f("MagneticFieldStrength"),
        })
    } else {
        None
    };

    Ok(Phantom { grid, perfusion, att, t1, t2, t2star, m0, dseg, fieldmap, labels, params })
}

/// One voxel of the T2' derivation table (spec P0 change 6), in milliseconds. Inputs in seconds
/// and already validated positive in the foreground.
fn t2prime_ms_of(t2_s: f32, t2star_s: f32) -> f32 {
    if t2star_s >= t2_s {
        f32::INFINITY
    } else {
        let rate = 1.0 / (t2star_s as f64) - 1.0 / (t2_s as f64);
        (1000.0 / rate) as f32
    }
}

impl Phantom {
    pub fn nvox(&self) -> usize {
        self.grid.dims.iter().product()
    }

    /// The per-voxel T2' map in milliseconds: `INFINITY` in the background (the compartments
    /// are zero there) and wherever `T2* >= T2`, else `1 / (1/T2* - 1/T2)`. Never zero, never
    /// NaN: the acquisition stage divides by it.
    pub fn t2prime_ms(&self) -> Vec<f32> {
        self.dseg
            .iter()
            .zip(self.t2.iter().zip(&self.t2star))
            .map(|(&l, (&t2, &t2s))| if l > 0 { t2prime_ms_of(t2, t2s) } else { f32::INFINITY })
            .collect()
    }

    /// The per-voxel T2 map in milliseconds, `INFINITY` in the background.
    pub fn t2_ms(&self) -> Vec<f32> {
        self.dseg.iter().zip(&self.t2).map(|(&l, &t2)| if l > 0 { t2 * 1000.0 } else { f32::INFINITY }).collect()
    }

    /// The constancy test: is `T2` and the derived `T2'` bitwise constant within every
    /// foreground label? Returns the offending `(label, voxel, map)` otherwise.
    fn constancy(&self) -> Result<(Vec<f32>, Vec<f32>), (i32, usize, &'static str)> {
        let t2p = self.t2prime_ms();
        let mut t2_ms = Vec::with_capacity(self.labels.len());
        let mut t2p_ms = Vec::with_capacity(self.labels.len());
        for (l, _) in &self.labels {
            let mut first: Option<(f32, f32)> = None;
            for (i, &lab) in self.dseg.iter().enumerate() {
                if lab != *l {
                    continue;
                }
                let here = (self.t2[i], t2p[i]);
                match first {
                    None => first = Some(here),
                    Some(f) => {
                        if f.0.to_bits() != here.0.to_bits() {
                            return Err((*l, i, "T2map"));
                        }
                        if f.1.to_bits() != here.1.to_bits() {
                            return Err((*l, i, "T2' (from T2map and T2starmap)"));
                        }
                    }
                }
            }
            let (t2, t2p) = first.expect("labels come from dseg, so each has a voxel");
            t2_ms.push(t2 * 1000.0);
            t2p_ms.push(t2p);
        }
        Ok((t2_ms, t2p_ms))
    }

    /// Resolve the relaxation representation. `Auto` takes `Class` when the constancy test
    /// passes and `Voxel` otherwise; `Class` is an error when it fails, naming the label, the
    /// voxel and the map.
    pub fn relaxation(&self, mode: T2Mode) -> Result<(Relaxation, T2Mode), String> {
        let voxel = || Relaxation::Voxel { t2_ms: self.t2_ms(), t2p_ms: self.t2prime_ms() };
        match mode {
            T2Mode::Voxel => Ok((voxel(), T2Mode::Voxel)),
            T2Mode::Auto => match self.constancy() {
                Ok((t2_ms, t2p_ms)) => Ok((Relaxation::Class { t2_ms, t2p_ms }, T2Mode::Class)),
                Err(_) => Ok((voxel(), T2Mode::Voxel)),
            },
            T2Mode::Class => match self.constancy() {
                Ok((t2_ms, t2p_ms)) => Ok((Relaxation::Class { t2_ms, t2p_ms }, T2Mode::Class)),
                Err((l, i, map)) => Err(format!(
                    "--t2-mode class: {map} is not constant within label {l} (first differing voxel {i}); \
                     use voxel mode for this phantom")),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrsim_acq::io::{write_3d, write_3d_i16};

    fn crop() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))
    }

    /// A scratch copy of the crop, so tests can perturb files.
    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("aslscan_phantom_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for e in std::fs::read_dir(crop()).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), d.join(e.file_name())).unwrap();
        }
        d
    }

    /// Rewrite one map of a scratch phantom with a perturbed voxel.
    fn perturb(dir: &Path, map: &str, ph: &Phantom, mut f: impl FnMut(&mut Vec<f32>)) {
        let mut v = match map {
            "T1map" => ph.t1.clone(),
            "T2map" => ph.t2.clone(),
            "T2starmap" => ph.t2star.clone(),
            "M0map" => ph.m0.clone(),
            _ => panic!(),
        };
        f(&mut v);
        write_3d(&dir.join(format!("{map}.nii.gz")), ph.grid.dims, &v, &ph.grid).unwrap();
    }

    fn first_voxel_of(ph: &Phantom, label: i32) -> usize {
        ph.dseg.iter().position(|&l| l == label).unwrap()
    }

    #[test]
    fn loads_the_crop_with_three_labels() {
        let ph = load(&crop()).unwrap();
        assert_eq!(ph.grid.dims, [24, 24, 6]);
        assert_eq!(ph.labels, vec![(1, "grey_matter".to_string()), (2, "white_matter".to_string()), (3, "csf".to_string())]);
        assert_eq!(ph.params, Some(PhantomParams { lambda: Some(0.9), t1b: Some(1.65), field_strength: Some(3.0) }));
        assert!(ph.fieldmap.is_none());
        let gm = first_voxel_of(&ph, 1);
        assert!((ph.t2[gm] - 0.08).abs() < 1e-6 && (ph.perfusion[gm] - 60.0).abs() < 1e-4);
        let csf = first_voxel_of(&ph, 3);
        assert!((ph.att[csf] - 1000.0).abs() < 1e-3, "the CSF sentinel survives conversion");
    }

    #[test]
    fn unit_strings_missing_maps_and_grid_mismatches_are_errors() {
        let d = scratch("units");
        std::fs::write(d.join("att.json"), "{\"Units\": \"ms\"}").unwrap();
        let e = load(&d).unwrap_err();
        assert!(e.contains("att") && e.contains("ms"), "{e}");
        std::fs::write(d.join("att.json"), "{\"Units\": \"s\"}").unwrap();
        std::fs::remove_file(d.join("T2starmap.nii.gz")).unwrap();
        assert!(load(&d).unwrap_err().contains("T2starmap"));
        let d = scratch("grid");
        let ph = load(&d).unwrap();
        let small = Grid { dims: [24, 24, 5], voxel_to_world: ph.grid.voxel_to_world };
        write_3d(&d.join("M0map.nii.gz"), small.dims, &vec![1.0; 24 * 24 * 5], &small).unwrap();
        let e = load(&d).unwrap_err();
        assert!(e.contains("M0map") && e.contains("[24, 24, 5]"), "{e}");
    }

    #[test]
    fn t2prime_derivation_follows_the_table() {
        let ph = load(&crop()).unwrap();
        let t2p = ph.t2prime_ms();
        let expect = |t2: f64, t2s: f64| (1000.0 / (1.0 / t2s - 1.0 / t2)) as f32;
        for (i, &l) in ph.dseg.iter().enumerate() {
            match l {
                0 => assert!(t2p[i].is_infinite()),
                1 => assert!((t2p[i] - expect(0.08, 0.066)).abs() < 1e-2, "GM {}", t2p[i]),
                2 => assert!((t2p[i] - expect(0.11, 0.053)).abs() < 1e-2, "WM {}", t2p[i]),
                3 => assert!((t2p[i] - expect(0.3, 0.2)).abs() < 1e-2, "CSF {}", t2p[i]),
                _ => unreachable!(),
            }
        }
        // T2* >= T2 is tolerated as no inhomogeneity decay
        let d = scratch("t2s_big");
        let wm = first_voxel_of(&ph, 2);
        perturb(&d, "T2starmap", &ph, |v| v[wm] = 0.12);
        let p2 = load(&d).unwrap();
        assert!(p2.t2prime_ms()[wm].is_infinite());
        // a foreground zero is an error naming the label and voxel
        let d = scratch("t2s_zero");
        perturb(&d, "T2starmap", &ph, |v| v[wm] = 0.0);
        let e = load(&d).unwrap_err();
        assert!(e.contains("T2starmap") && e.contains(&format!("voxel {wm}")) && e.contains("label 2"), "{e}");
    }

    #[test]
    fn auto_resolves_to_class_on_the_crop() {
        let ph = load(&crop()).unwrap();
        let (r, mode) = ph.relaxation(T2Mode::Auto).unwrap();
        assert_eq!(mode, T2Mode::Class);
        match r {
            Relaxation::Class { t2_ms, t2p_ms } => {
                assert_eq!(t2_ms.len(), 3);
                assert!((t2_ms[0] - 80.0).abs() < 1e-3 && (t2_ms[1] - 110.0).abs() < 1e-3 && (t2_ms[2] - 300.0).abs() < 1e-3);
                assert!(t2p_ms.iter().all(|v| v.is_finite() && *v > 0.0));
            }
            other => panic!("{other:?}"),
        }
        let (r, mode) = ph.relaxation(T2Mode::Voxel).unwrap();
        assert_eq!(mode, T2Mode::Voxel);
        assert!(matches!(r, Relaxation::Voxel { .. }));
    }

    #[test]
    fn one_perturbed_t2_voxel_falls_back_to_voxel_and_fails_class() {
        let ph = load(&crop()).unwrap();
        let gm = first_voxel_of(&ph, 1);
        let d = scratch("t2_pert");
        perturb(&d, "T2map", &ph, |v| v[gm] = 0.0801);
        let p2 = load(&d).unwrap();
        assert_eq!(p2.relaxation(T2Mode::Auto).unwrap().1, T2Mode::Voxel);
        let e = p2.relaxation(T2Mode::Class).unwrap_err();
        assert!(e.contains("label 1") && e.contains("T2map"), "{e}");
    }

    #[test]
    fn t1_perturbation_does_not_affect_the_mode() {
        let ph = load(&crop()).unwrap();
        let gm = first_voxel_of(&ph, 1);
        let d = scratch("t1_pert");
        perturb(&d, "T1map", &ph, |v| v[gm] = 1.5);
        assert_eq!(load(&d).unwrap().relaxation(T2Mode::Auto).unwrap().1, T2Mode::Class);
    }

    #[test]
    fn t2star_perturbation_that_keeps_t2prime_constant_is_still_class() {
        // A tiny phantom with T2 = 0.1 and T2* in {0.15, 0.2} within one label: T2' is INFINITY
        // throughout, so the derived map is constant even though T2* is not.
        let d = std::env::temp_dir().join(format!("aslscan_phantom_tiny_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let grid = Grid { dims: [4, 4, 2], voxel_to_world: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };
        let n = 32;
        let seg: Vec<i16> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let fg = |v: f32| -> Vec<f32> { seg.iter().map(|&s| if s > 0 { v } else { 0.0 }).collect() };
        let t2s: Vec<f32> = (0..n).map(|i| if seg[i] == 0 { 0.0 } else if i % 4 == 0 { 0.15 } else { 0.2 }).collect();
        for (name, data, unit) in [
            ("perfusion", fg(60.0), "ml/100g/min"), ("att", fg(0.8), "s"), ("T1map", fg(1.3), "s"),
            ("T2map", fg(0.1), "s"), ("T2starmap", t2s, "s"), ("M0map", fg(70.0), "arbitrary"),
        ] {
            write_3d(&d.join(format!("{name}.nii.gz")), grid.dims, &data, &grid).unwrap();
            std::fs::write(d.join(format!("{name}.json")), format!("{{\"Units\": \"{unit}\"}}")).unwrap();
        }
        write_3d_i16(&d.join("dseg.nii.gz"), grid.dims, &seg, &grid).unwrap();
        std::fs::write(d.join("dseg.json"), "{\"Units\": \"label indices\", \"LabelMap\": {\"1\": \"tissue\"}}").unwrap();
        let ph = load(&d).unwrap();
        assert_eq!(ph.labels, vec![(1, "tissue".to_string())]);
        let (r, mode) = ph.relaxation(T2Mode::Auto).unwrap();
        assert_eq!(mode, T2Mode::Class);
        match r {
            Relaxation::Class { t2_ms, t2p_ms } => {
                assert!((t2_ms[0] - 100.0).abs() < 1e-3);
                assert!(t2p_ms[0].is_infinite());
            }
            other => panic!("{other:?}"),
        }
    }
}
