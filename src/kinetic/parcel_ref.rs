//! A brute-force reference for depleted reads (P7 plan, Task 2), test-only. It integrates the
//! label parcel by parcel over arrival time and shares no code with the closed forms in
//! `kinetic` beyond [`Kinetic`]: each parcel is labeled, carried, inverted by the suppression
//! pulses whose region it is in, exchanged with probability `1 - exp(-s/tau_ex)` and depleted by
//! every excitation after it entered the slab, and the integral is taken by Gauss-Legendre on the
//! pieces between the integrand's jumps.

use super::{Kinetic, LabelType};

/// Where a suppression pulse acts on a parcel (P4 part D's table, restated per parcel).
#[derive(Debug, Clone, Copy)]
pub enum Region {
    Global,
    Slab(f64),
    Arrival,
}

/// One voxel and one read.
#[derive(Debug, Clone)]
pub struct Case {
    pub k: Kinetic,
    /// Perfusion (ml/100g/min), ATT (s), tissue T1 (s), M0.
    pub f: f64,
    pub att: f64,
    pub t1t: f64,
    pub m0: f64,
    /// The read time (s from the start of labeling).
    pub t: f64,
    /// Every excitation before the read: `(time, flip in degrees)`.
    pub excitations: Vec<(f64, f64)>,
    /// A parcel enters the slab `entry_lead` before it arrives (`ATT - d`; 0: at arrival).
    pub entry_lead: f64,
    /// Suppression pulses (sorted), their inversion efficiency, and where they act.
    pub pulses: Vec<f64>,
    pub epsilon: f64,
    pub region: Region,
    pub tau_ex: Option<f64>,
}

/// The 8-point Gauss-Legendre nodes and weights on `[-1, 1]`.
const GL: [(f64, f64); 8] = [
    (-0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
    (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (-0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (-0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
];

/// `integral_lo^hi g(u) du` with `g` smooth on every piece between `breaks`: each piece split
/// into `sub` equal parts, 8-point Gauss-Legendre on each.
fn integrate(lo: f64, hi: f64, breaks: &[f64], sub: usize, g: impl Fn(f64) -> f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    let mut pts: Vec<f64> = breaks.iter().copied().filter(|&b| b > lo && b < hi).collect();
    pts.push(lo);
    pts.push(hi);
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut s = 0.0;
    for w in pts.windows(2) {
        let h = (w[1] - w[0]) / sub as f64;
        for j in 0..sub {
            let (a, b) = (w[0] + h * j as f64, w[0] + h * (j + 1) as f64);
            let (c, r) = (0.5 * (a + b), 0.5 * (b - a));
            s += GL.iter().map(|&(x, wt)| wt * g(c + r * x)).sum::<f64>() * r;
        }
    }
    s
}

impl Case {
    fn m0b(&self) -> f64 {
        self.m0 / self.k.lambda
    }

    /// `T1'`: the tissue T1 with the flow's outflow, the GKM's residue time.
    fn t1p(&self) -> f64 {
        1.0 / (1.0 / self.t1t + (self.f / 6000.0) / self.k.lambda)
    }

    /// The parcel's sub-bolus coordinate (its labeling time for (P)CASL, its arrival less ATT for
    /// PASL) and whether pulse `p` inverts it, by the region's rule.
    fn inside(&self, u: f64, p: f64, att: f64) -> bool {
        let a = u - att;
        match (self.region, self.k.label_type) {
            (Region::Global, LabelType::Pasl) => true,
            (Region::Global, _) => p >= a,
            (Region::Slab(d), _) => p >= a + d,
            (Region::Arrival, _) => p >= u,
        }
    }

    fn suppression(&self, u: f64, att: f64) -> f64 {
        self.pulses.iter().filter(|&&p| self.inside(u, p, att)).map(|_| 1.0 - 2.0 * self.epsilon).product()
    }

    /// Depletion of a parcel that entered the slab at `entry`, read at `t`.
    fn depletion(&self, entry: f64) -> f64 {
        self.excitations
            .iter()
            .filter(|&&(e, _)| entry <= e && e < self.t)
            .map(|&(_, a)| a.to_radians().cos())
            .product()
    }

    /// The label delivered per unit arrival time at `u`, before relaxation in the voxel.
    fn delivered(&self, u: f64) -> f64 {
        let k = &self.k;
        let carried = match k.label_type {
            // labeled at u - ATT, carried ATT in blood
            LabelType::Casl | LabelType::Pcasl => (-self.att / k.t1b).exp(),
            // labeled at 0, carried u in blood
            LabelType::Pasl => (-u / k.t1b).exp(),
        };
        2.0 * k.alpha * self.m0b() * (self.f / 6000.0) * carried
    }

    fn breaks(&self, att: f64, lead: f64) -> Vec<f64> {
        let mut b: Vec<f64> = self.excitations.iter().map(|&(e, _)| e + lead).collect();
        for &p in &self.pulses {
            match (self.region, self.k.label_type) {
                (Region::Global, LabelType::Pasl) => {}
                (Region::Global, _) => b.push(p + att),
                (Region::Slab(d), _) => b.push(p + att - d),
                (Region::Arrival, _) => b.push(p),
            }
        }
        b
    }

    /// `(intravascular, total)` of the label read at `t`, every factor applied, before `sin(a)`.
    pub fn read(&self, sub: usize) -> (f64, f64) {
        let (lo, hi) = (self.att, (self.att + self.k.tau).min(self.t));
        let t1p = self.t1p();
        let breaks = self.breaks(self.att, self.entry_lead);
        let parcel = |u: f64| {
            self.delivered(u)
                * (-(self.t - u) / t1p).exp()
                * self.suppression(u, self.att)
                * self.depletion(u - self.entry_lead)
        };
        let total = integrate(lo, hi, &breaks, sub, parcel);
        let iv = match self.tau_ex {
            None => total,
            Some(te) => integrate(lo, hi, &breaks, sub, |u| parcel(u) * (-(self.t - u) / te).exp()),
        };
        (iv, total)
    }

    /// The arterial read at `t` before `sin(a)` and survival: the one parcel in the voxel's
    /// arteries, entered into the slab `entry_lead_a` before `t`.
    pub fn arterial(&self, abv: f64, aatt: f64, entry_lead_a: f64) -> f64 {
        let k = &self.k;
        let t = self.t;
        if !(aatt <= t && t < aatt + k.tau) {
            return 0.0;
        }
        let carried = match k.label_type {
            LabelType::Casl | LabelType::Pcasl => (-aatt / k.t1b).exp(),
            LabelType::Pasl => (-t / k.t1b).exp(),
        };
        2.0 * k.alpha * self.m0b() * abv * carried * self.suppression(t, aatt) * self.depletion(t - entry_lead_a)
    }
}
