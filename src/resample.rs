//! Overlap-weighted box averaging between axis-aligned grids that share an origin corner.
//!
//! The phantom, acquisition and simulation grids are all axis-aligned and start at the same
//! corner (spec: grids and resampling), so the 3D overlap of a target voxel with the source
//! voxels is separable: one weight list per axis, multiplied together. Each target cell's
//! weights are the fraction of THAT target cell each source cell covers, so a fully covered
//! target sums to 1 and a partially covered edge cell (target extending past the source) sums to
//! less: a partial voxel, not a rescaled one.
//!
//! What is averaged is magnetization, never model parameters: `kinetic` and `mrsignal` run on
//! the phantom grid and their outputs come through [`Resampler::mean`]. The exceptions are
//! named as such: [`Resampler::rate_mean`] for `voxel`-mode relaxation maps (rates, M0-weighted),
//! [`Resampler::masked_mean`] for the ground-truth ATT (perfused voxels only), and
//! [`Resampler::majority`] for the segmentation.

use mrsim_acq::grid::Grid;

/// Per-axis overlap weights: for target cell `j`, `per_target[j]` lists `(source index, weight)`.
#[derive(Debug, Clone)]
pub struct AxisWeights {
    pub per_target: Vec<Vec<(usize, f64)>>,
}

/// Overlap weights for one axis. Source cells are `[i*d_src, (i+1)*d_src)`, target cells
/// `[j*d_dst, (j+1)*d_dst)`, both from the shared corner at 0. Overlaps below `1e-9 * d_dst` are
/// dropped as floating-point noise.
pub fn axis_weights(n_src: usize, d_src: f64, n_dst: usize, d_dst: f64) -> AxisWeights {
    axis_weights_offset(n_src, d_src, n_dst, d_dst, 0.0)
}

/// [`axis_weights`] with the target grid's corner `offset` mm from the source's along the axis
/// (target cells `[offset + j*d_dst, offset + (j+1)*d_dst)`). A target cell hanging past either
/// source edge gets the partial weight of what it does cover, as at the far edge.
pub fn axis_weights_offset(n_src: usize, d_src: f64, n_dst: usize, d_dst: f64, offset: f64) -> AxisWeights {
    assert!(d_src > 0.0 && d_dst > 0.0, "voxel sizes must be positive");
    assert!(offset.is_finite(), "grid offset must be finite");
    let eps = 1e-9 * d_dst;
    let mut per_target = Vec::with_capacity(n_dst);
    for j in 0..n_dst {
        let (lo, hi) = (offset + j as f64 * d_dst, offset + (j + 1) as f64 * d_dst);
        if hi <= 0.0 {
            per_target.push(Vec::new());
            continue;
        }
        let i0 = (lo / d_src).floor().max(0.0) as usize;
        let i1 = ((hi / d_src).ceil() as usize).min(n_src);
        let mut w = Vec::new();
        for i in i0..i1 {
            let (slo, shi) = (i as f64 * d_src, (i + 1) as f64 * d_src);
            let overlap = hi.min(shi) - lo.max(slo);
            if overlap > eps {
                w.push((i, overlap / d_dst));
            }
        }
        per_target.push(w);
    }
    AxisWeights { per_target }
}

/// A source-grid to target-grid box resampler.
#[derive(Debug, Clone)]
pub struct Resampler {
    pub x: AxisWeights,
    pub y: AxisWeights,
    pub z: AxisWeights,
    pub src_dims: [usize; 3],
    pub dst_dims: [usize; 3],
}

impl Resampler {
    pub fn new(src_dims: [usize; 3], src_vox: [f64; 3], dst_dims: [usize; 3], dst_vox: [f64; 3]) -> Self {
        Self::with_offset(src_dims, src_vox, dst_dims, dst_vox, [0.0; 3])
    }

    /// A resampler whose target corner sits `offset` mm (per axis, along the index direction)
    /// from the source corner; [`corner_offset`] gives it for two grids.
    pub fn with_offset(src_dims: [usize; 3], src_vox: [f64; 3], dst_dims: [usize; 3], dst_vox: [f64; 3], offset: [f64; 3])
        -> Self
    {
        Resampler {
            x: axis_weights_offset(src_dims[0], src_vox[0], dst_dims[0], dst_vox[0], offset[0]),
            y: axis_weights_offset(src_dims[1], src_vox[1], dst_dims[1], dst_vox[1], offset[1]),
            z: axis_weights_offset(src_dims[2], src_vox[2], dst_dims[2], dst_vox[2], offset[2]),
            src_dims,
            dst_dims,
        }
    }

    #[inline]
    fn sat(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.src_dims[0] * (y + self.src_dims[1] * z)
    }

    /// Visit every (source index, product weight) pair of target voxel `(x, y, z)`.
    fn for_each_overlap(&self, x: usize, y: usize, z: usize, mut f: impl FnMut(usize, f64)) {
        for &(sz, wz) in &self.z.per_target[z] {
            for &(sy, wy) in &self.y.per_target[y] {
                let wyz = wy * wz;
                for &(sx, wx) in &self.x.per_target[x] {
                    f(self.sat(sx, sy, sz), wx * wyz);
                }
            }
        }
    }

    fn map(&self, mut cell: impl FnMut(usize, usize, usize) -> f32) -> Vec<f32> {
        let [nx, ny, nz] = self.dst_dims;
        let mut out = vec![0.0f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    out[x + nx * (y + ny * z)] = cell(x, y, z);
                }
            }
        }
        out
    }

    /// Volume-weighted mean: `sum(w * v)` over the overlapped source cells (weights sum to the
    /// covered fraction of the target cell).
    pub fn mean(&self, src: &[f32]) -> Vec<f32> {
        assert_eq!(src.len(), self.src_dims.iter().product::<usize>(), "source is not on the source grid");
        self.map(|x, y, z| {
            let mut acc = 0.0f64;
            self.for_each_overlap(x, y, z, |i, w| acc += w * src[i] as f64);
            acc as f32
        })
    }

    /// Mean of `src` over the overlapped source cells where `mask` holds; `0` where none do.
    pub fn masked_mean(&self, src: &[f32], mask: &[bool]) -> Vec<f32> {
        assert_eq!(src.len(), mask.len());
        self.map(|x, y, z| {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            self.for_each_overlap(x, y, z, |i, w| {
                if mask[i] {
                    num += w * src[i] as f64;
                    den += w;
                }
            });
            if den > 0.0 { (num / den) as f32 } else { 0.0 }
        })
    }

    /// Weighted mean of the RATES `1 / src_time` with weights `w`, inverted back to a time.
    /// `INFINITY` in means rate 0; `INFINITY` out where the weighted rate or the total weight is
    /// zero. This is the `voxel`-mode single-exponential stand-in for a mixed voxel (spec: grids
    /// and resampling); it is exact where the target cell is homogeneous.
    pub fn rate_mean(&self, src_time: &[f32], w: &[f32]) -> Vec<f32> {
        assert_eq!(src_time.len(), w.len());
        self.map(|x, y, z| {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            self.for_each_overlap(x, y, z, |i, ov| {
                let wi = ov * w[i] as f64;
                let t = src_time[i] as f64;
                let rate = if t.is_infinite() { 0.0 } else { 1.0 / t };
                num += wi * rate;
                den += wi;
            });
            if den > 0.0 && num > 0.0 { (den / num) as f32 } else { f32::INFINITY }
        })
    }

    /// Majority vote over the overlapped labels by overlap weight; ties go to the lower label.
    pub fn majority(&self, src: &[i32]) -> Vec<i32> {
        let [nx, ny, nz] = self.dst_dims;
        let mut out = vec![0i32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mut tally: Vec<(i32, f64)> = Vec::new();
                    self.for_each_overlap(x, y, z, |i, w| {
                        match tally.iter_mut().find(|(l, _)| *l == src[i]) {
                            Some(e) => e.1 += w,
                            None => tally.push((src[i], w)),
                        }
                    });
                    let mut best: Option<(i32, f64)> = None;
                    for &(l, w) in &tally {
                        best = match best {
                            None => Some((l, w)),
                            Some((bl, bw)) if w > bw || (w == bw && l < bl) => Some((l, w)),
                            b => b,
                        };
                    }
                    out[x + nx * (y + ny * z)] = best.map_or(0, |b| b.0);
                }
            }
        }
        out
    }

    /// The source z-cells (and weights) that target slice `z` overlaps. Per-slice timing is
    /// evaluated over exactly this slab.
    pub fn z_slab(&self, z: usize) -> &[(usize, f64)] {
        &self.z.per_target[z]
    }

    /// Volume-weighted mean of a source field into target slice `z` only, the field given as a
    /// closure over the flat source index so a slice-dependent quantity (kinetics at that
    /// slice's timing) need not be materialised as a whole phantom volume. Returns the
    /// `dst_dims[0] * dst_dims[1]` target slice, layout `x + nx*y`.
    pub fn mean_slice(&self, z: usize, src: impl Fn(usize) -> f64) -> Vec<f32> {
        let [nx, ny, _] = self.dst_dims;
        let mut out = vec![0.0f32; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let mut acc = 0.0f64;
                self.for_each_overlap(x, y, z, |i, w| acc += w * src(i));
                out[x + nx * y] = acc as f32;
            }
        }
        out
    }
}

/// The phantom grid's voxel sizes, requiring an axis-aligned affine: every off-diagonal entry of
/// the linear part must be zero to `1e-9`, or the box average would silently rotate the object.
pub fn axis_aligned_voxels(g: &Grid) -> Result<[f64; 3], String> {
    let m = &g.voxel_to_world;
    for (r, row) in m.iter().enumerate().take(3) {
        for (c, &v) in row.iter().enumerate().take(3) {
            if r != c && v.abs() > 1e-9 {
                return Err(format!(
                    "phantom affine is not axis-aligned: entry [{r}][{c}] = {v}; the box resampler \
                     is defined on axis-aligned grids only"));
            }
        }
    }
    Ok([m[0][0].abs(), m[1][1].abs(), m[2][2].abs()])
}

/// Where the acquisition grid sits on the phantom's field of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GridOrigin {
    /// The grids share their outer corner (P1).
    #[default]
    Corner,
    /// Acquisition voxel 0 is centred on phantom voxel 0, as simasl's
    /// `transform_resample_affine` places it (P2 addendum, part A).
    VoxelCentre,
}

impl GridOrigin {
    pub fn parse(s: &str) -> Result<GridOrigin, String> {
        match s {
            "corner" => Ok(GridOrigin::Corner),
            "voxel-centre" => Ok(GridOrigin::VoxelCentre),
            other => Err(format!("grid_origin {other:?}: expected \"corner\" or \"voxel-centre\"")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            GridOrigin::Corner => "corner",
            GridOrigin::VoxelCentre => "voxel-centre",
        }
    }
}

/// Per axis, how far `dst`'s outer corner sits from `src`'s, in mm along the index direction
/// (both grids axis-aligned with the same axis directions). Zero for P1's corner-aligned grids.
pub fn corner_offset(src: &Grid, dst: &Grid) -> Result<[f64; 3], String> {
    let (sv, dv) = (axis_aligned_voxels(src)?, axis_aligned_voxels(dst)?);
    let mut off = [0.0f64; 3];
    for a in 0..3 {
        let (s, d) = (src.voxel_to_world[a][a], dst.voxel_to_world[a][a]);
        if s.signum() != d.signum() {
            return Err(format!("grids disagree on the direction of axis {a}"));
        }
        let corner_s = src.voxel_to_world[a][3] - 0.5 * s;
        let corner_d = dst.voxel_to_world[a][3] - 0.5 * d;
        off[a] = (corner_d - corner_s) * s.signum();
        // Exact zero for corner-aligned grids, so the P1 weights are reproduced bit for bit.
        if off[a].abs() <= 1e-9 * sv[a].min(dv[a]) {
            off[a] = 0.0;
        }
    }
    Ok(off)
}

/// The acquisition grid for a phantom's field of view: `ceil(extent / voxel)` cells per axis
/// (an in-plane `matrix_override` replaces the first two), with the phantom's axis directions,
/// placed by `origin`. Under `Corner` voxel `j`'s centre sits at `corner + (j + 0.5) * voxel`;
/// under `VoxelCentre` at `o + j * voxel`, `o` being phantom voxel 0's centre.
pub fn acquisition_grid(phantom: &Grid, voxel_mm: [f64; 3], matrix_override: Option<[usize; 2]>, origin: GridOrigin)
    -> Result<Grid, String>
{
    let pv = axis_aligned_voxels(phantom)?;
    let mut dims = [0usize; 3];
    for a in 0..3 {
        if voxel_mm[a] <= 0.0 {
            return Err(format!("acquisition voxel size along axis {a} must be positive"));
        }
        let extent = phantom.dims[a] as f64 * pv[a];
        dims[a] = (extent / voxel_mm[a] - 1e-9).ceil().max(1.0) as usize;
    }
    if let Some([nx, ny]) = matrix_override {
        dims[0] = nx;
        dims[1] = ny;
    }
    let mut m = [[0.0f64; 4]; 4];
    m[3][3] = 1.0;
    for a in 0..3 {
        let sign = if phantom.voxel_to_world[a][a] < 0.0 { -1.0 } else { 1.0 };
        m[a][a] = sign * voxel_mm[a];
        m[a][3] = match origin {
            // phantom voxel 0 is centred at o; its edge is o - 0.5 * p; the new voxel 0 is
            // centred half a new voxel in from that edge.
            GridOrigin::Corner => phantom.voxel_to_world[a][3] - 0.5 * phantom.voxel_to_world[a][a] + 0.5 * m[a][a],
            GridOrigin::VoxelCentre => phantom.voxel_to_world[a][3],
        };
    }
    Ok(Grid { dims, voxel_to_world: m })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }

    #[test]
    fn weights_tile_exactly_when_grids_nest() {
        let w = axis_weights(8, 1.0, 4, 2.0);
        for j in 0..4 {
            assert_eq!(w.per_target[j].len(), 2);
            assert_eq!(w.per_target[j][0].0, 2 * j);
            assert_eq!(w.per_target[j][1].0, 2 * j + 1);
            assert!(approx(w.per_target[j][0].1, 0.5) && approx(w.per_target[j][1].1, 0.5));
        }
    }

    #[test]
    fn weights_handle_non_integer_ratios() {
        let w = axis_weights(3, 1.0, 2, 1.5);
        assert_eq!(w.per_target[0].len(), 2);
        assert!(approx(w.per_target[0][0].1, 2.0 / 3.0) && w.per_target[0][0].0 == 0);
        assert!(approx(w.per_target[0][1].1, 1.0 / 3.0) && w.per_target[0][1].0 == 1);
        assert!(approx(w.per_target[1][0].1, 1.0 / 3.0) && w.per_target[1][0].0 == 1);
        assert!(approx(w.per_target[1][1].1, 2.0 / 3.0) && w.per_target[1][1].0 == 2);
        for j in 0..2 {
            let s: f64 = w.per_target[j].iter().map(|p| p.1).sum();
            assert!(approx(s, 1.0));
        }
    }

    #[test]
    fn padding_gives_a_partial_last_voxel() {
        let w = axis_weights(5, 1.0, 3, 2.0);
        assert_eq!(w.per_target[2], vec![(4, 0.5)]);
    }

    #[test]
    fn constant_stays_constant_where_covered() {
        let r = Resampler::new([6, 6, 4], [1.0, 1.0, 1.0], [3, 2, 2], [2.0, 3.0, 2.0]);
        let src = vec![7.0f32; 6 * 6 * 4];
        let dst = r.mean(&src);
        assert!(dst.iter().all(|&v| (v - 7.0).abs() < 1e-5));
    }

    #[test]
    fn mass_is_conserved_when_grids_tile_the_same_extent() {
        let (sd, sv, dd, dv) = ([6, 9, 4], [1.0, 1.0, 1.5], [3, 3, 2], [2.0, 3.0, 3.0]);
        let r = Resampler::new(sd, sv, dd, dv);
        let src: Vec<f32> = (0..6 * 9 * 4).map(|i| ((i * 37) % 11) as f32 + 0.5).collect();
        let dst = r.mean(&src);
        let m_src: f64 = src.iter().map(|&v| v as f64).sum::<f64>() * sv.iter().product::<f64>();
        let m_dst: f64 = dst.iter().map(|&v| v as f64).sum::<f64>() * dv.iter().product::<f64>();
        assert!((m_src - m_dst).abs() <= 1e-6 * m_src, "{m_src} vs {m_dst}");
    }

    #[test]
    fn boundary_voxel_is_the_weighted_mean_of_signals_not_of_parameters() {
        use crate::mrsignal::tissue_se;
        // Two source cells, m0 = 1, T1 = 1 s and 3 s, TR 4 s, into one target cell.
        let r = Resampler::new([2, 1, 1], [1.0, 1.0, 1.0], [1, 1, 1], [2.0, 1.0, 1.0]);
        let sig = [tissue_se(1.0, 1.0, 4.0) as f32, tissue_se(1.0, 3.0, 4.0) as f32];
        let got = r.mean(&sig)[0] as f64;
        let want = 0.5 * (1.0 - (-4.0f64).exp()) + 0.5 * (1.0 - (-4.0f64 / 3.0).exp());
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        let of_mean_params = tissue_se(1.0, 2.0, 4.0);
        assert!((got - of_mean_params).abs() > 1e-3, "the two must differ: {got} vs {of_mean_params}");
    }

    #[test]
    fn rate_mean_matches_a_hand_value_and_is_infinity_for_zero_weight() {
        let r = Resampler::new([2, 1, 1], [1.0, 1.0, 1.0], [1, 1, 1], [2.0, 1.0, 1.0]);
        // times 100 and 50 ms with weights 3 and 1: rate = (3/100 + 1/50)/4 = 0.0125 -> 80 ms
        let got = r.rate_mean(&[100.0, 50.0], &[3.0, 1.0])[0];
        assert!((got - 80.0).abs() < 1e-4, "{got}");
        assert!(r.rate_mean(&[100.0, 50.0], &[0.0, 0.0])[0].is_infinite());
        // an INFINITY time is a zero rate and pulls the mean toward INFINITY, never NaN
        let got = r.rate_mean(&[f32::INFINITY, 50.0], &[1.0, 1.0])[0];
        assert!((got - 100.0).abs() < 1e-4, "{got}");
        assert!(r.rate_mean(&[f32::INFINITY, f32::INFINITY], &[1.0, 1.0])[0].is_infinite());
    }

    #[test]
    fn majority_breaks_ties_low() {
        let r = Resampler::new([4, 1, 1], [1.0, 1.0, 1.0], [1, 1, 1], [4.0, 1.0, 1.0]);
        assert_eq!(r.majority(&[2, 2, 1, 1])[0], 1);
        assert_eq!(r.majority(&[3, 3, 3, 1])[0], 3);
        assert_eq!(r.majority(&[0, 0, 0, 2])[0], 0);
    }

    #[test]
    fn acquisition_grid_covers_the_fov() {
        let ph = Grid {
            dims: [24, 24, 6],
            voxel_to_world: [[1.0, 0.0, 0.0, -10.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
        };
        let g = acquisition_grid(&ph, [3.0, 3.0, 3.0], None, GridOrigin::Corner).unwrap();
        assert_eq!(g.dims, [8, 8, 2]);
        let g2 = acquisition_grid(&ph, [3.5, 3.5, 3.0], None, GridOrigin::Corner).unwrap();
        assert_eq!(g2.dims, [7, 7, 2]);
        // corner is at -10 - 0.5 = -10.5; new voxel 0 centre at corner + 1.75
        assert!((g2.voxel_to_world[0][3] - (-10.5 + 1.75)).abs() < 1e-12);
        assert!((g2.voxel_to_world[0][0] - 3.5).abs() < 1e-12);
        let g3 = acquisition_grid(&ph, [3.0, 3.0, 3.0], Some([16, 12]), GridOrigin::Corner).unwrap();
        assert_eq!(g3.dims, [16, 12, 2]);
        // a negative axis keeps its direction
        let mut flipped = ph.clone();
        flipped.voxel_to_world[0][0] = -1.0;
        let g4 = acquisition_grid(&flipped, [2.0, 2.0, 2.0], None, GridOrigin::Corner).unwrap();
        assert!((g4.voxel_to_world[0][0] + 2.0).abs() < 1e-12);
        assert!((g4.voxel_to_world[0][3] - (-10.0 + 0.5 - 1.0)).abs() < 1e-12);
        // oblique is refused
        let mut obl = ph.clone();
        obl.voxel_to_world[0][1] = 0.1;
        assert!(acquisition_grid(&obl, [3.0, 3.0, 3.0], None, GridOrigin::Corner).is_err());
    }

    #[test]
    fn voxel_centre_origin_centres_voxel_0_on_the_phantom_voxel_0() {
        let ph = Grid {
            dims: [24, 24, 6],
            voxel_to_world: [[1.0, 0.0, 0.0, -10.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
        };
        let g = acquisition_grid(&ph, [3.0, 2.0, 1.5], None, GridOrigin::VoxelCentre).unwrap();
        assert_eq!(g.dims, [8, 12, 4]);
        for a in 0..3 {
            assert_eq!(g.voxel_to_world[a][3], ph.voxel_to_world[a][3]);
        }
        // corner offsets: (0.5 p - 0.5 v) per axis
        let off = corner_offset(&ph, &g).unwrap();
        assert!(approx(off[0], -1.0) && approx(off[1], -0.5) && approx(off[2], -0.25), "{off:?}");
        assert_eq!(corner_offset(&ph, &acquisition_grid(&ph, [3.0, 2.0, 1.5], None, GridOrigin::Corner).unwrap()).unwrap(), [0.0; 3]);
        // a negative axis keeps its direction and still centres voxel 0 on voxel 0
        let mut flipped = ph.clone();
        flipped.voxel_to_world[0][0] = -1.0;
        let g = acquisition_grid(&flipped, [3.0, 2.0, 1.5], None, GridOrigin::VoxelCentre).unwrap();
        assert_eq!(g.voxel_to_world[0][0], -3.0);
        assert_eq!(g.voxel_to_world[0][3], -10.0);
        assert!(approx(corner_offset(&flipped, &g).unwrap()[0], -1.0));
        assert_eq!(GridOrigin::parse("voxel-centre").unwrap(), GridOrigin::VoxelCentre);
        assert_eq!(GridOrigin::parse("corner").unwrap(), GridOrigin::Corner);
        assert!(GridOrigin::parse("center").is_err());
    }

    #[test]
    fn simasl_matrix_reproduces_from_the_derived_voxel_size() {
        // the ASLDRO phantoms: 197 x 233 x 189 at 1 mm; acq_matrix [64, 64, 12]
        let ph = Grid {
            dims: [197, 233, 189],
            voxel_to_world: [[1.0, 0.0, 0.0, -98.0], [0.0, 1.0, 0.0, -134.0], [0.0, 0.0, 1.0, -72.0], [0.0, 0.0, 0.0, 1.0]],
        };
        for origin in [GridOrigin::Corner, GridOrigin::VoxelCentre] {
            let g = acquisition_grid(&ph, [197.0 / 64.0, 233.0 / 64.0, 189.0 / 12.0], None, origin).unwrap();
            assert_eq!(g.dims, [64, 64, 12]);
            let g = acquisition_grid(&ph, [1.0, 1.0, 1.0], None, origin).unwrap();
            assert_eq!(g.dims, [197, 233, 189]);
        }
    }

    #[test]
    fn offset_weights_give_partial_cells_at_the_near_edge() {
        // target cells of 3 from -1: [-1, 2), [2, 5), [5, 8) over 6 unit source cells
        let w = axis_weights_offset(6, 1.0, 3, 3.0, -1.0);
        assert_eq!(w.per_target[0].iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1]);
        let s0: f64 = w.per_target[0].iter().map(|p| p.1).sum();
        assert!(approx(s0, 2.0 / 3.0), "{s0}");
        let s1: f64 = w.per_target[1].iter().map(|p| p.1).sum();
        assert!(approx(s1, 1.0));
        assert_eq!(w.per_target[2].iter().map(|p| p.0).collect::<Vec<_>>(), vec![5]);
        // wholly before the source: empty
        assert!(axis_weights_offset(6, 1.0, 1, 1.0, -2.0).per_target[0].is_empty());
        // offset 0 is the P1 weights exactly
        let (a, b) = (axis_weights(7, 1.0, 3, 2.5), axis_weights_offset(7, 1.0, 3, 2.5, 0.0));
        for j in 0..3 {
            assert_eq!(a.per_target[j], b.per_target[j]);
        }
        // the identity grid at offset 0 is the identity
        let r = Resampler::with_offset([4, 3, 2], [1.0; 3], [4, 3, 2], [1.0; 3], [0.0; 3]);
        let src: Vec<f32> = (0..24).map(|i| i as f32 * 0.37 + 1.0).collect();
        assert_eq!(r.mean(&src), src);
    }

    #[test]
    fn z_slab_lists_the_overlapped_source_slices() {
        let r = Resampler::new([4, 4, 6], [1.0, 1.0, 1.0], [2, 2, 2], [2.0, 2.0, 3.0]);
        assert_eq!(r.z_slab(1).iter().map(|p| p.0).collect::<Vec<_>>(), vec![3, 4, 5]);
    }
}
