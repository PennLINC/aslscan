//! The Buxton general kinetic model, per voxel, as simasl's `GkmFilter` computes it
//! (`src/asldro/filters/gkm_filter.py`), so that the two can be diffed. Seconds throughout.
//!
//! Faithful means faithful to the guards too: the delivery-state masks (`gkm_filter.py:157-161`),
//! every `np.divide(.., out=zeros, where=..)` site, and the two *different* tests on a zero
//! arterial T1 (`> 0` in the PASL branch at `:205`, `!= 0` in the CASL/PCASL branch at `:252`).
//! The fixture diff at 1e-9 relative (`tests/fixtures/gkm.txt`) is the referee, and its exact
//! zeros come from masks, so the Rust reproduces them as exact zeros rather than as tiny values.

/// Which labeling scheme the series uses. Selects the kinetic branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelType {
    Pasl,
    Casl,
    Pcasl,
}

/// Per-row kinetic constants, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kinetic {
    pub label_type: LabelType,
    /// Bolus duration `tau` (s): `LabelingDuration` for (P)CASL, `BolusCutOffDelayTime[0]` for
    /// PASL.
    pub tau: f64,
    /// Label efficiency `alpha`.
    pub alpha: f64,
    /// Blood-brain partition coefficient (ml/g).
    pub lambda: f64,
    /// Arterial blood T1 (s).
    pub t1b: f64,
}

/// simasl's `np.divide(n, d, out=zeros, where=d != 0)`: zero where the denominator is zero.
#[inline]
fn div0(n: f64, d: f64) -> f64 {
    if d != 0.0 {
        n / d
    } else {
        0.0
    }
}

/// `delta_m` for one voxel at signal time `t` (s), the difference in longitudinal magnetization
/// the label delivers.
///
/// `f_ml_100g_min` is the perfusion rate exactly as the phantom stores it; the `/ 6000` to
/// per-second happens here (`gkm_filter.py:105`). `dt` is the arterial transit time, `t1t` the
/// tissue T1, `m0` the tissue equilibrium magnetization. The result is what simasl's masked
/// array holds for this voxel: `0` where the bolus has not arrived (`0 < t <= dt`), and also `0`
/// for `t <= 0`, where simasl's chained comparison makes every mask false and the zero-initialised
/// output is never written.
pub fn delta_m(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64) -> f64 {
    let (f, m0b, t1p) = gkm_constants(k, f_ml_100g_min, t1t, m0);
    gkm_body(k, f, m0b, dt, t1p, t)
}

/// `f` (per second), `M0b` and `T1'`, with simasl's guards.
#[inline]
fn gkm_constants(k: &Kinetic, f_ml_100g_min: f64, t1t: f64, m0: f64) -> (f64, f64, f64) {
    let f = f_ml_100g_min / 6000.0;
    // M0b and f/lambda, with simasl's scalar guard on lambda (gkm_filter.py:135-147).
    let (m0b, flow_over_lambda) =
        if k.lambda != 0.0 { (m0 / k.lambda, f / k.lambda) } else { (0.0, 0.0) };
    // 1/T1' = 1/T1t + f/lambda, both divisions guarded (gkm_filter.py:149-153).
    let denom = div0(1.0, t1t) + flow_over_lambda;
    let t1p = div0(1.0, denom);
    (f, m0b, t1p)
}

/// [`delta_m`] with `T1'` given rather than derived from the tissue `T1` (P4 addendum, part A:
/// the intravascular part is the GKM with `T1'` replaced by `T1''`). The same delivery masks,
/// branches and guards; the arithmetic of [`delta_m`] is this function's.
pub fn delta_m_with_t1p(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1p: f64, m0: f64, t: f64) -> f64 {
    let f = f_ml_100g_min / 6000.0;
    let m0b = if k.lambda != 0.0 { m0 / k.lambda } else { 0.0 };
    gkm_body(k, f, m0b, dt, t1p, t)
}

#[inline]
fn gkm_body(k: &Kinetic, f: f64, m0b: f64, dt: f64, t1p: f64, t: f64) -> f64 {
    // Delivery state. simasl builds three masks and writes the arriving and arrived results
    // into a zero-initialised array; the not-arrived mask only ever writes zero.
    let arriving = dt < t && t < dt + k.tau;
    let arrived = t >= dt + k.tau;
    if !(arriving || arrived) {
        return 0.0;
    }

    match k.label_type {
        LabelType::Pasl => {
            // k = 1/T1b - 1/T1', with the scalar guard on T1b (gkm_filter.py:168-170).
            let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
            // The arterial decay factor is replaced by ZERO (not the quotient) for T1b <= 0
            // (gkm_filter.py:205, :218): `exp(-t/T1b) if t1b > 0 else 0`.
            let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
            if arriving {
                // q_pasl_arriving, numerator and denominator computed separately so that
                // t == dt cannot divide by zero (it is masked out anyway) (gkm_filter.py:173-185).
                let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * t).exp());
                let q = div0(num, kk * (t - dt));
                2.0 * m0b * f * (t - dt) * k.alpha * decay * q
            } else {
                let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * (dt + k.tau)).exp());
                let q = div0(num, kk * k.tau);
                2.0 * m0b * f * k.alpha * k.tau * decay * q
            }
        }
        LabelType::Casl | LabelType::Pcasl => {
            // Here the T1b test is `!= 0`, not `> 0` (gkm_filter.py:252, :265).
            let decay = if k.t1b != 0.0 { (-dt / k.t1b).exp() } else { 0.0 };
            if arriving {
                let q = 1.0 - (-div0(t - dt, t1p)).exp();
                2.0 * m0b * f * t1p * k.alpha * decay * q
            } else {
                let q = 1.0 - (-div0(k.tau, t1p)).exp();
                2.0 * m0b * f * t1p * k.alpha * decay * (-div0(t - k.tau - dt, t1p)).exp() * q
            }
        }
    }
}

/// The sub-bolus of parcels `a..b` (in `[0, tau]`; P4 addendum, "the one idea"): (P)CASL
/// parcels labeled in `[a, b]` are a GKM bolus of duration `b - a` started `a` later (the
/// transit decay is the same for every parcel); PASL parcels arriving in `[ATT + a, ATT + b]`
/// are a GKM bolus with `ATT' = ATT + a` and no shift. The uncut sub-bolus (`a = 0`,
/// `b = tau`) calls [`delta_m`] with the original arguments, so the P1 path is the P1 code.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_sub(k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, a: f64, b: f64) -> f64 {
    if a == 0.0 && b == k.tau {
        return delta_m(k, f, dt, t1t, m0, t);
    }
    let ks = Kinetic { tau: b - a, ..*k };
    match k.label_type {
        LabelType::Pasl => delta_m(&ks, f, dt + a, t1t, m0, t),
        LabelType::Casl | LabelType::Pcasl => delta_m(&ks, f, dt, t1t, m0, t - a),
    }
}

/// `T1''` of the intravascular part: `1/T1'' = 1/T1' + 1/tau_ex`, with `T1'` guarded as
/// [`delta_m`] guards it. `None` where `T1'` is the guarded zero, which only happens where
/// `f/lambda` is zero and the GKM is therefore zero too.
fn t1pp(k: &Kinetic, f_ml_100g_min: f64, t1t: f64, m0: f64, tau_ex: f64) -> Option<(f64, f64, f64)> {
    let (f, m0b, t1p) = gkm_constants(k, f_ml_100g_min, t1t, m0);
    if t1p == 0.0 {
        return None;
    }
    Some((f, m0b, 1.0 / (1.0 / t1p + 1.0 / tau_ex)))
}

/// The intravascular part of [`delta_m`] (P4 addendum, part A): each parcel resident for `s`
/// has not yet exchanged with probability `exp(-s/tau_ex)`, which makes it the GKM with `T1'`
/// replaced by `T1''`. (P)CASL is [`delta_m_with_t1p`]'s arithmetic. PASL is not: its branch
/// forms `exp(kk t)` and `exp(-kk dt)` separately, and with `T1''` short `kk` is large and
/// negative, so those overflow to `0 * (inf - inf)`; here the exponents are combined, which is
/// the same quantity and finite for every `tau_ex`.
pub fn delta_m_iv(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64, tau_ex: f64) -> f64 {
    let Some((f, m0b, t1pp)) = t1pp(k, f_ml_100g_min, t1t, m0, tau_ex) else { return 0.0 };
    match k.label_type {
        LabelType::Casl | LabelType::Pcasl => gkm_body(k, f, m0b, dt, t1pp, t),
        LabelType::Pasl => {
            // The part never exceeds the whole delta_m computes. Where delta_m's own
            // kk = 1/T1b - 1/T1' is zero (its quotient guard) or within rounding of it (its
            // difference of exponentials cancels), delta_m is zero or noise; it stays as simasl
            // computes it, and T1'' moves kk off zero, so unclamped the part would be a positive
            // share of a zero whole and the extravascular rest negative.
            let iv = pasl_stable(k, f, m0b, dt, t1pp, t);
            iv.min(delta_m(k, f_ml_100g_min, dt, t1t, m0, t)).max(0.0)
        }
    }
}

/// [`delta_m_iv`] of the sub-bolus `a..b`, by the shifts of [`delta_m_sub`].
#[allow(clippy::too_many_arguments)]
pub fn delta_m_iv_sub(k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, a: f64, b: f64, tau_ex: f64) -> f64 {
    if a == 0.0 && b == k.tau {
        return delta_m_iv(k, f, dt, t1t, m0, t, tau_ex);
    }
    let ks = Kinetic { tau: b - a, ..*k };
    match k.label_type {
        LabelType::Pasl => delta_m_iv(&ks, f, dt + a, t1t, m0, t, tau_ex),
        LabelType::Casl | LabelType::Pcasl => delta_m_iv(&ks, f, dt, t1t, m0, t - a, tau_ex),
    }
}

/// The PASL branch of [`gkm_body`] with `exp(kk t) (exp(-kk x) - exp(-kk y))` evaluated as
/// `exp(kk (t - y)) expm1(kk (y - x))`: the same masks and guards. Combining the exponents keeps
/// it finite at large `|kk|`; `expm1` keeps it accurate at small `|kk|`, where `tau_ex` makes
/// `T1''` nearly `T1b` and the difference of two exponentials would cancel to noise.
fn pasl_stable(k: &Kinetic, f: f64, m0b: f64, dt: f64, t1p: f64, t: f64) -> f64 {
    let arriving = dt < t && t < dt + k.tau;
    let arrived = t >= dt + k.tau;
    if !(arriving || arrived) {
        return 0.0;
    }
    let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
    let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
    if arriving {
        let num = (kk * (t - dt)).exp_m1();
        let q = div0(num, kk * (t - dt));
        2.0 * m0b * f * (t - dt) * k.alpha * decay * q
    } else {
        let num = (kk * (t - dt - k.tau)).exp() * (kk * k.tau).exp_m1();
        let q = div0(num, kk * k.tau);
        2.0 * m0b * f * k.alpha * k.tau * decay * q
    }
}

/// The arterial (macrovascular) difference magnetization (P4 addendum, part B), with the
/// parcel factor `g = 1`: `2 alpha M0b aBV exp(-aATT/T1b)` for (P)CASL and
/// `2 alpha M0b aBV exp(-t/T1b)` for PASL, inside `aATT <= t < aATT + tau`, zero outside, with
/// the GKM's `lambda` and `T1b` guards. Returns the value and the sub-bolus coordinate of the
/// parcel the voxel's arteries hold (`t - aATT` for both labeling types), `None` outside.
pub fn arterial_dm(k: &Kinetic, abv: f64, aatt: f64, m0: f64, t: f64) -> (f64, Option<f64>) {
    if !(aatt <= t && t < aatt + k.tau) {
        return (0.0, None);
    }
    let m0b = if k.lambda != 0.0 { m0 / k.lambda } else { 0.0 };
    let decay = match k.label_type {
        LabelType::Pasl => if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 },
        LabelType::Casl | LabelType::Pcasl => if k.t1b != 0.0 { (-aatt / k.t1b).exp() } else { 0.0 },
    };
    (2.0 * k.alpha * m0b * abv * decay, Some(t - aatt))
}

/// The delta-M at `t` of the label that arrived in the voxel during `[u1, u2)` (P6 addendum,
/// part B: a Look-Locker readout depletes the label that has arrived, so the delta-M read splits
/// by arrival window). Each parcel arriving at `u` relaxes with the GKM's residue
/// `exp(-(t - u)/T1')` after it; the windows partition `[0, t)`, and over them the sum is
/// [`delta_m`] at `t`.
///
/// (P)CASL delivers `2 alpha M0b f exp(-dt/T1b)` on `[dt, dt + tau]`:
/// `2 alpha M0b f exp(-dt/T1b) T1' (exp(-(t - u_hi)/T1') - exp(-(t - u_lo)/T1'))`. PASL delivers
/// `2 alpha M0b f exp(-u/T1b)` on `[dt, dt + tau]`: with `q = 1/T1b - 1/T1'`,
/// `2 alpha M0b f exp(-t/T1b) exp(q t) (exp(-q u_lo) - exp(-q u_hi)) / q`, zero at `q = 0` as the
/// GKM's guard makes it. `u_lo = max(u1, dt)`, `u_hi = min(u2, dt + tau, t)`, zero when
/// `u_hi <= u_lo`; the GKM's guards on `lambda`, `T1'` and `T1b` hold.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_arrival(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64, u1: f64, u2: f64) -> f64 {
    let (f, m0b, t1p) = gkm_constants(k, f_ml_100g_min, t1t, m0);
    let (u_lo, u_hi) = (u1.max(dt), u2.min(dt + k.tau).min(t));
    if u_hi <= u_lo || t1p == 0.0 {
        return 0.0;
    }
    match k.label_type {
        LabelType::Pasl => {
            let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
            let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
            let num = (kk * t).exp() * ((-kk * u_lo).exp() - (-kk * u_hi).exp());
            2.0 * m0b * f * k.alpha * decay * div0(num, kk)
        }
        LabelType::Casl | LabelType::Pcasl => {
            let decay = if k.t1b != 0.0 { (-dt / k.t1b).exp() } else { 0.0 };
            2.0 * m0b * f * t1p * k.alpha * decay * ((-(t - u_hi) / t1p).exp() - (-(t - u_lo) / t1p).exp())
        }
    }
}

/// The delta-M a Look-Locker readout reads, before its `sin(a)` (P6 addendum, part B): `e` are
/// this slice's excitation times of the cycle's readouts up to and including the one read (s from
/// the start of labeling, increasing), `flips_deg` the flips of the readouts before it. The label
/// that arrived in `[e_k, e_k+1)` (`e_0 = 0`) has been excited by every later readout before this
/// one, each leaving `cos(a)` of it:
/// `sum_k prod_{m > k, m < n} cos(a_m) * delta_m_arrival(e_n; e_k, e_k+1)`.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_read(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, e: &[f64], flips_deg: &[f64]) -> f64 {
    let n = e.len();
    assert!(n >= 1 && flips_deg.len() + 1 == n, "{n} excitations need {} earlier flips, got {}", n.saturating_sub(1), flips_deg.len());
    let t = e[n - 1];
    (0..n)
        .map(|w| {
            let lo = if w == 0 { 0.0 } else { e[w - 1] };
            let depletion: f64 = flips_deg[w..].iter().map(|a| a.to_radians().cos()).product();
            depletion * delta_m_arrival(k, f_ml_100g_min, dt, t1t, m0, t, lo, e[w])
        })
        .sum()
}

// ---- P7 part A: the P4 parts under Look-Locker; part C: depletion from slab entry ----

/// The intravascular part of [`delta_m_arrival`] (P7 addendum, part A): the label of the window
/// `[u1, u2)` not yet exchanged at `t`, the arrival form with `T1'` replaced by
/// `T1'' = (1/T1' + 1/tau_ex)^-1` (P4 part A's residue). PASL combines its exponents as
/// [`delta_m_iv`]'s `pasl_stable` does, `exp(kk (t - u_hi)) expm1(kk (u_hi - u_lo))`, which stays
/// finite where a short `tau_ex` makes `kk` large and negative, and is clamped to
/// `[0, delta_m_arrival(same window)]` as P4 clamps the whole bolus. (P)CASL is the residue
/// difference with `T1''`, which never exceeds the whole.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_arrival_iv(
    k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64, u1: f64, u2: f64, tau_ex: f64,
) -> f64 {
    let Some((f, m0b, t1pp)) = t1pp(k, f_ml_100g_min, t1t, m0, tau_ex) else { return 0.0 };
    let (u_lo, u_hi) = (u1.max(dt), u2.min(dt + k.tau).min(t));
    if u_hi <= u_lo || t1pp == 0.0 {
        return 0.0;
    }
    match k.label_type {
        LabelType::Pasl => {
            let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1pp);
            let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
            let num = (kk * (t - u_hi)).exp() * (kk * (u_hi - u_lo)).exp_m1();
            let iv = 2.0 * m0b * f * k.alpha * decay * div0(num, kk);
            iv.min(delta_m_arrival(k, f_ml_100g_min, dt, t1t, m0, t, u1, u2)).max(0.0)
        }
        LabelType::Casl | LabelType::Pcasl => {
            let decay = if k.t1b != 0.0 { (-dt / k.t1b).exp() } else { 0.0 };
            2.0 * m0b * f * t1pp * k.alpha * decay * ((-(t - u_hi) / t1pp).exp() - (-(t - u_lo) / t1pp).exp())
        }
    }
}

/// [`delta_m_arrival`] of the sub-bolus `a..b` of `[0, tau]`, by [`delta_m_sub`]'s shifts:
/// (P)CASL a bolus of length `b - a` read at `t - a`, its arrival window shifted by `-a` (a parcel
/// labeled at `l` arrives at `l + dt`); PASL the arrival delay `dt + a`, the window unshifted. The
/// uncut sub-bolus calls [`delta_m_arrival`] itself.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_arrival_sub(
    k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, u1: f64, u2: f64, a: f64, b: f64,
) -> f64 {
    if a == 0.0 && b == k.tau {
        return delta_m_arrival(k, f, dt, t1t, m0, t, u1, u2);
    }
    let ks = Kinetic { tau: b - a, ..*k };
    match k.label_type {
        LabelType::Pasl => delta_m_arrival(&ks, f, dt + a, t1t, m0, t, u1, u2),
        LabelType::Casl | LabelType::Pcasl => delta_m_arrival(&ks, f, dt, t1t, m0, t - a, u1 - a, u2 - a),
    }
}

/// [`delta_m_arrival_iv`] of the sub-bolus `a..b`, by the same shifts as [`delta_m_arrival_sub`].
#[allow(clippy::too_many_arguments)]
pub fn delta_m_arrival_iv_sub(
    k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, u1: f64, u2: f64, a: f64, b: f64, tau_ex: f64,
) -> f64 {
    if a == 0.0 && b == k.tau {
        return delta_m_arrival_iv(k, f, dt, t1t, m0, t, u1, u2, tau_ex);
    }
    let ks = Kinetic { tau: b - a, ..*k };
    match k.label_type {
        LabelType::Pasl => delta_m_arrival_iv(&ks, f, dt + a, t1t, m0, t, u1, u2, tau_ex),
        LabelType::Casl | LabelType::Pcasl => delta_m_arrival_iv(&ks, f, dt, t1t, m0, t - a, u1 - a, u2 - a, tau_ex),
    }
}

/// The label a depleted read sees, split as P4 splits it: `iv` the label not yet exchanged (the
/// blood compartment), `ev` the exchanged rest (the tissue compartment). Without exchange all of
/// it is `iv` and `ev` is zero (P4: all label intravascular).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReadParts {
    pub iv: f64,
    pub ev: f64,
}

impl ReadParts {
    pub fn total(&self) -> f64 {
        self.iv + self.ev
    }
}

/// One window's contribution, every sub-bolus with its factor: `(iv, whole)`.
#[allow(clippy::too_many_arguments)]
fn window_parts(
    k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, lo: f64, hi: f64, subs: &[(f64, f64, f64)],
    tau_ex: Option<f64>,
) -> (f64, f64) {
    let (mut iv, mut whole) = (0.0, 0.0);
    for &(a, b, fac) in subs {
        let w = delta_m_arrival_sub(k, f, dt, t1t, m0, t, lo, hi, a, b);
        whole += fac * w;
        iv += fac * match tau_ex {
            Some(te) => delta_m_arrival_iv_sub(k, f, dt, t1t, m0, t, lo, hi, a, b, te),
            None => w,
        };
    }
    (iv, whole)
}

/// The arrival window `w` of a depleted read with slab-entry shift `delta` (P7 addendum, part C):
/// `[e_{w-1} + delta, e_w + delta)`, the first starting at `0`. A parcel arriving in it entered
/// the slab before every excitation from `w` on, so they all deplete it; `delta = 0` is P6's
/// arrival in the voxel.
fn window(e: &[f64], w: usize, delta: f64) -> (f64, f64) {
    (if w == 0 { 0.0 } else { e[w - 1] + delta }, e[w] + delta)
}

/// The depleted read at the last of the excitations `e` (s from the start of labeling,
/// increasing), with every P4 part (P7 addendum, parts A and C): the sub-boli `subs` as
/// `(a, b, factor)` (one `(0, tau, 1)` without bolus-position suppression or Hadamard), the
/// intravascular split when `tau_ex` is given, and the windows shifted by `delta = ATT - d` for
/// depletion from slab entry (`0` in 2D). `flips_deg` are the earlier excitations', as in
/// [`delta_m_read`], which this reduces to (one full sub-bolus, no exchange, `delta = 0`).
/// Direct: O(n) per read. [`delta_m_read_all`] gives every read of a train in O(n) in all.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_read_parts(
    k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, e: &[f64], flips_deg: &[f64], subs: &[(f64, f64, f64)],
    tau_ex: Option<f64>, delta: f64,
) -> ReadParts {
    let n = e.len();
    assert!(n >= 1 && flips_deg.len() + 1 == n, "{n} excitations need {} earlier flips, got {}", n.saturating_sub(1), flips_deg.len());
    let t = e[n - 1];
    let (mut iv, mut whole) = (0.0, 0.0);
    for w in 0..n {
        let (lo, hi) = window(e, w, delta);
        let depletion: f64 = flips_deg[w..].iter().map(|a| a.to_radians().cos()).product();
        let (i, h) = window_parts(k, f, dt, t1t, m0, t, lo, hi, subs, tau_ex);
        iv += depletion * i;
        whole += depletion * h;
    }
    match tau_ex {
        Some(_) => ReadParts { iv, ev: whole - iv },
        None => ReadParts { iv: whole, ev: 0.0 },
    }
}

/// One component of [`delta_m_read_all`]: the depleted sum at every excitation, where `term(t,
/// lo, hi)` is the component's window contribution at `t` and decays as `exp(-t/t1x)` once the
/// window has fully arrived (`hi <= t`). The full windows are carried as one sum, decayed and
/// depleted per step; each window is evaluated directly once, when it becomes full, and the one
/// window straddling `t` is evaluated directly at each read. The depletion of window `w` at read
/// `n` is `exp(lp[n] - lp[w])`, `lp` the prefix sums of `ln cos(a)` (finite: flips are in
/// `(0, 90]`).
fn read_all_component(e: &[f64], lp: &[f64], delta: f64, t1x: f64, term: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
    let n = e.len();
    let mut out = vec![0.0; n];
    if t1x == 0.0 {
        // the guarded zero T1': every term is zero (the GKM's guard)
        return out;
    }
    let (mut s, mut full) = (0.0f64, 0usize);
    for r in 0..n {
        let t = e[r];
        if r > 0 {
            s *= (lp[r] - lp[r - 1]).exp() * (-(t - e[r - 1]) / t1x).exp();
        }
        while full <= r && window(e, full, delta).1 <= t {
            let (lo, hi) = window(e, full, delta);
            s += (lp[r] - lp[full]).exp() * term(t, lo, hi);
            full += 1;
        }
        let mut v = s;
        if full <= r {
            let (lo, hi) = window(e, full, delta);
            v += (lp[r] - lp[full]).exp() * term(t, lo, hi);
        }
        out[r] = v;
    }
    out
}

/// [`delta_m_read_parts`] at every excitation of a train in O(n) in all (P7 plan, Task 1): `e`
/// the excitations and `flips_deg` every excitation's flip (the last is not read). Every full
/// window's term is proportional to `exp(-t/T1')` (`exp(-t/T1'')` for the intravascular part):
/// for (P)CASL directly, for PASL as `exp(-t/T1b) exp(kk t)`. The recursion is carried relative to
/// the previous excitation, so no exponent spans more than one step.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_read_all(
    k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, e: &[f64], flips_deg: &[f64], subs: &[(f64, f64, f64)],
    tau_ex: Option<f64>, delta: f64,
) -> Vec<ReadParts> {
    let n = e.len();
    assert_eq!(flips_deg.len(), n, "one flip per excitation");
    let mut lp = vec![0.0; n + 1];
    for (m, a) in flips_deg.iter().enumerate() {
        lp[m + 1] = lp[m] + a.to_radians().cos().ln();
    }
    let (_, _, t1p) = gkm_constants(k, f, t1t, m0);
    let whole = read_all_component(e, &lp, delta, t1p, |t, lo, hi| {
        subs.iter().map(|&(a, b, fac)| fac * delta_m_arrival_sub(k, f, dt, t1t, m0, t, lo, hi, a, b)).sum()
    });
    match tau_ex {
        None => whole.into_iter().map(|w| ReadParts { iv: w, ev: 0.0 }).collect(),
        Some(te) => {
            let t1x = t1pp(k, f, t1t, m0, te).map_or(0.0, |(_, _, x)| x);
            let iv = read_all_component(e, &lp, delta, t1x, |t, lo, hi| {
                subs.iter()
                    .map(|&(a, b, fac)| fac * delta_m_arrival_iv_sub(k, f, dt, t1t, m0, t, lo, hi, a, b, te))
                    .sum()
            });
            iv.into_iter().zip(whole).map(|(i, w)| ReadParts { iv: i, ev: w - i }).collect()
        }
    }
}

/// The arterial read at excitation `e_n` before its `sin(a)` (P7 addendum, parts A and C):
/// [`arterial_dm`] times `factor` (the crushing survival and the parcel's bolus-position factor,
/// which the caller evaluates at the parcel `e_n - aATT`), times `cos(a_m)` of every earlier
/// excitation in `[e_n - delta_a, e_n)`: the parcel entered the slab `delta_a = aATT - d` before
/// it is read. `delta_a = 0` is part A's fresh arterial blood.
#[allow(clippy::too_many_arguments)]
pub fn arterial_read(
    k: &Kinetic, abv: f64, aatt: f64, m0: f64, e_n: f64, earlier: &[(f64, f64)], delta_a: f64, factor: f64,
) -> f64 {
    let (v, _) = arterial_dm(k, abv, aatt, m0, e_n);
    if v == 0.0 {
        return 0.0;
    }
    let depletion: f64 = earlier
        .iter()
        .filter(|&&(e_m, _)| e_n - delta_a <= e_m && e_m < e_n)
        .map(|&(_, a)| a.to_radians().cos())
        .product();
    v * factor * depletion
}

#[cfg(test)]
pub(crate) mod parcel_ref;

#[cfg(test)]
mod tests {
    use super::*;

    const K_PCASL: Kinetic =
        Kinetic { label_type: LabelType::Pcasl, tau: 1.8, alpha: 0.85, lambda: 0.9, t1b: 1.65 };
    const K_PASL: Kinetic =
        Kinetic { label_type: LabelType::Pasl, tau: 0.7, alpha: 0.98, lambda: 0.9, t1b: 1.65 };
    // GM in the ASLDRO 3T phantom.
    const F: f64 = 60.0;
    const DT: f64 = 0.8;
    const T1T: f64 = 1.33;
    const M0: f64 = 74.622;

    fn t1_prime(f_ml: f64, t1t: f64, lambda: f64) -> f64 {
        1.0 / (1.0 / t1t + (f_ml / 6000.0) / lambda)
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs())
    }

    #[test]
    fn not_arrived_is_exactly_zero() {
        for t in [0.5, DT, 0.0, -1.0] {
            assert_eq!(delta_m(&K_PCASL, F, DT, T1T, M0, t), 0.0, "t = {t}");
            assert_eq!(delta_m(&K_PASL, F, DT, T1T, M0, t), 0.0, "t = {t}");
        }
    }

    #[test]
    fn pcasl_arrived_matches_closed_form() {
        // The simasl default: PLD 1.8 + tau 1.8.
        let t = 3.6;
        let t1p = t1_prime(F, T1T, 0.9);
        let m0b = M0 / 0.9;
        let f = F / 6000.0;
        let want = 2.0 * m0b * f * t1p * 0.85 * (-DT / 1.65).exp()
            * (-(t - 1.8 - DT) / t1p).exp()
            * (1.0 - (-1.8 / t1p).exp());
        let got = delta_m(&K_PCASL, F, DT, T1T, M0, t);
        assert!(close(got, want, 1e-12), "{got} vs {want}");
        assert!(got > 0.0);
    }

    #[test]
    fn pcasl_arriving_at_the_bolus_midpoint() {
        let t = DT + 0.9;
        let t1p = t1_prime(F, T1T, 0.9);
        let want = 2.0 * (M0 / 0.9) * (F / 6000.0) * t1p * 0.85 * (-DT / 1.65).exp()
            * (1.0 - (-(t - DT) / t1p).exp());
        assert!(close(delta_m(&K_PCASL, F, DT, T1T, M0, t), want, 1e-12));
    }

    #[test]
    fn pasl_arriving_and_arrived_match_the_k_form() {
        let t1p = t1_prime(F, T1T, 0.9);
        let kk = 1.0 / 1.65 - 1.0 / t1p;
        let m0b = M0 / 0.9;
        let f = F / 6000.0;
        // arriving: dt < 1.2 < dt + 0.7
        let t = 1.2;
        let q = (kk * t).exp() * ((-kk * DT).exp() - (-kk * t).exp()) / (kk * (t - DT));
        let want = 2.0 * m0b * f * (t - DT) * 0.98 * (-t / 1.65).exp() * q;
        assert!(close(delta_m(&K_PASL, F, DT, T1T, M0, t), want, 1e-12));
        // arrived: 2.0 >= dt + 0.7
        let t = 2.0;
        let q = (kk * t).exp() * ((-kk * DT).exp() - (-kk * (DT + 0.7)).exp()) / (kk * 0.7);
        let want = 2.0 * m0b * f * 0.98 * 0.7 * (-t / 1.65).exp() * q;
        assert!(close(delta_m(&K_PASL, F, DT, T1T, M0, t), want, 1e-12));
    }

    #[test]
    fn zero_perfusion_is_zero_everywhere() {
        for t in [0.5, 1.2, 2.0, 3.6, 10.0] {
            assert_eq!(delta_m(&K_PCASL, 0.0, DT, T1T, M0, t), 0.0);
            assert_eq!(delta_m(&K_PASL, 0.0, DT, T1T, M0, t), 0.0);
        }
    }

    #[test]
    fn zero_t1b_is_zero_in_both_branches() {
        // simasl replaces the whole arterial exponential with zero, not the quotient
        // (gkm_filter.py:205, :252), so this is 0, not the exp(0) = 1 a guarded division gives.
        let pc = Kinetic { t1b: 0.0, ..K_PCASL };
        let pa = Kinetic { t1b: 0.0, ..K_PASL };
        assert_eq!(delta_m(&pc, F, DT, T1T, M0, 3.6), 0.0);
        assert_eq!(delta_m(&pa, F, DT, T1T, M0, 2.0), 0.0);
        assert_eq!(delta_m(&pa, F, DT, T1T, M0, 1.2), 0.0);
    }

    #[test]
    fn pasl_k_exactly_zero_takes_the_quotient_guard_not_the_limit() {
        // Fidelity, not physics: with 1/T1b == 1/T1' exactly, k == 0, every q denominator is
        // zero, and simasl's np.divide(where=) yields q = 0, hence delta_m = 0 -- not the finite
        // 0/0 limit a textbook would take. Choose t1b so the subtraction is exactly zero.
        let t1p = t1_prime(F, T1T, 0.9);
        let k = Kinetic { t1b: 1.0 / (1.0 / t1p), ..K_PASL };
        let kk = 1.0 / k.t1b - 1.0 / t1p;
        assert_eq!(kk, 0.0, "the test premise: k must be exactly zero");
        assert_eq!(delta_m(&k, F, DT, T1T, M0, 1.2), 0.0);
        assert_eq!(delta_m(&k, F, DT, T1T, M0, 2.0), 0.0);
        // and the intravascular part follows the whole: zero, not a positive part of a zero
        // (final review: 0.00098 at exchange_time 0.5, every sub-bolus too)
        for tau_ex in [0.5, 3.0] {
            for t in [1.2, 2.0] {
                assert_eq!(delta_m_iv(&k, F, DT, T1T, M0, t, tau_ex), 0.0, "tau_ex {tau_ex} t {t}");
                assert_eq!(delta_m_iv_sub(&k, F, DT, T1T, M0, t, 0.1, 0.4, tau_ex), 0.0);
            }
        }
    }

    #[test]
    fn pasl_intravascular_part_never_exceeds_a_cancelled_whole() {
        // Near kk = 0 but not at it, delta_m's difference of exponentials cancels to zero or
        // noise (re-review: T1b = 2.9032258064516134 with T1t = 3 gives kk ~ -5.6e-17,
        // delta_m = 0 and an unclamped part of 0.0013). The part stays within [0, delta_m]
        // for the whole and every sub-bolus, a few ULPs and a little further either side.
        let (t1t, m0, t1b0) = (3.0, 1.0, 2.9032258064516134);
        let k0 = Kinetic { label_type: LabelType::Pasl, tau: 0.7, alpha: 0.85, lambda: 0.9, t1b: t1b0 };
        assert_eq!(delta_m(&k0, F, DT, t1t, m0, 2.0), 0.0, "the test premise: the whole cancels to zero");
        assert_eq!(delta_m_iv(&k0, F, DT, t1t, m0, 2.0, 0.5), 0.0);
        let mut steps: Vec<f64> = (-20..=20).map(|j| j as f64).collect();
        steps.extend([-1e6, -1e3, 1e3, 1e6]);
        for s in steps {
            let k = Kinetic { t1b: f64::from_bits((t1b0.to_bits() as i64 + s as i64) as u64), ..k0 };
            for tau_ex in [0.05, 0.5, 3.0] {
                for t in [1.0, 1.4, 2.0, 3.0] {
                    let dm = delta_m(&k, F, DT, t1t, m0, t);
                    let iv = delta_m_iv(&k, F, DT, t1t, m0, t, tau_ex);
                    assert!(iv >= 0.0 && iv <= dm.max(0.0), "t1b {} tau_ex {tau_ex} t {t}: iv {iv} dm {dm}", k.t1b);
                    for (a, b) in [(0.0, 0.3), (0.3, 0.7), (0.1, 0.4)] {
                        let dm = delta_m_sub(&k, F, DT, t1t, m0, t, a, b);
                        let iv = delta_m_iv_sub(&k, F, DT, t1t, m0, t, a, b, tau_ex);
                        assert!(iv >= 0.0 && iv <= dm.max(0.0), "sub {a}..{b} t {t}: iv {iv} dm {dm}");
                    }
                }
            }
        }
    }

    #[test]
    fn zero_tissue_t1_with_zero_flow_gives_zero_not_nan() {
        // t1p becomes 0 (guarded 1/0), every /t1p is guarded to 0, exp(-0) = 1, q = 0.
        let got = delta_m(&K_PCASL, 0.0, DT, 0.0, M0, 3.6);
        assert_eq!(got, 0.0);
        assert!(!delta_m(&K_PCASL, F, DT, 0.0, M0, 3.6).is_nan());
    }

    /// Fixture cases from simasl's GkmFilter (tools/gen_gkm_fixtures.py). Plain text, pure std.
    struct Case {
        k: Kinetic,
        t: f64,
        f: Vec<f64>,
        dt: Vec<f64>,
        t1t: Vec<f64>,
        m0: Vec<f64>,
        want: Vec<f64>,
    }

    fn parse_fixtures(text: &str) -> Vec<Case> {
        let mut cases = Vec::new();
        let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let floats = |l: &str| -> Vec<f64> {
            l.split_whitespace().skip(1).map(|t| t.parse::<f64>().unwrap()).collect()
        };
        while let Some(head) = lines.next() {
            let toks: Vec<&str> = head.split_whitespace().collect();
            assert_eq!(toks[0], "case", "unexpected line: {head}");
            let scalars = lines.next().unwrap();
            let s: Vec<&str> = scalars.split_whitespace().collect();
            assert_eq!(s[0], "scalars");
            let label_type = match s[1] {
                "pasl" => LabelType::Pasl,
                "casl" => LabelType::Casl,
                "pcasl" => LabelType::Pcasl,
                other => panic!("unknown label type {other}"),
            };
            let num = |i: usize| s[i].parse::<f64>().unwrap();
            let k = Kinetic { label_type, tau: num(2), alpha: num(4), lambda: num(5), t1b: num(6) };
            let t = num(3);
            let mut take = |name: &str| -> Vec<f64> {
                let l = lines.next().unwrap();
                assert!(l.starts_with(name), "expected {name}, got {l}");
                floats(l)
            };
            let f = take("perfusion_rate");
            let dt = take("transit_time");
            let t1t = take("t1_tissue");
            let m0 = take("m0");
            let want = take("delta_m");
            cases.push(Case { k, t, f, dt, t1t, m0, want });
        }
        cases
    }

    #[test]
    fn matches_simasl_gkm_fixtures_at_1e9() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gkm.txt"))
            .expect("tests/fixtures/gkm.txt is committed; regenerate with tools/gen_gkm_fixtures.py");
        let cases = parse_fixtures(&text);
        assert!(cases.len() >= 8, "too few cases: {}", cases.len());
        for (ci, c) in cases.iter().enumerate() {
            for i in 0..c.want.len() {
                let got = delta_m(&c.k, c.f[i], c.dt[i], c.t1t[i], c.m0[i], c.t);
                let want = c.want[i];
                if want == 0.0 {
                    assert_eq!(got, 0.0, "case {ci} voxel {i}: expected an exact zero, got {got:e}");
                } else {
                    let tol = 1e-9 * want.abs();
                    assert!((got - want).abs() <= tol,
                            "case {ci} voxel {i}: {got:.17e} vs {want:.17e} (rel {:.3e})",
                            (got - want).abs() / want.abs());
                }
            }
        }
    }

    // ------------------------------------------------------------------ P4

    /// `delta_m` exactly as it was before the P4 refactor (`p2-complete`): the bit-identity reference.
    fn delta_m_ref(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64) -> f64 {
        let f = f_ml_100g_min / 6000.0;
        // M0b and f/lambda, with simasl's scalar guard on lambda (gkm_filter.py:135-147).
        let (m0b, flow_over_lambda) =
            if k.lambda != 0.0 { (m0 / k.lambda, f / k.lambda) } else { (0.0, 0.0) };
        // 1/T1' = 1/T1t + f/lambda, both divisions guarded (gkm_filter.py:149-153).
        let denom = div0(1.0, t1t) + flow_over_lambda;
        let t1p = div0(1.0, denom);

        // Delivery state. simasl builds three masks and writes the arriving and arrived results
        // into a zero-initialised array; the not-arrived mask only ever writes zero.
        let arriving = dt < t && t < dt + k.tau;
        let arrived = t >= dt + k.tau;
        if !(arriving || arrived) {
            return 0.0;
        }

        match k.label_type {
            LabelType::Pasl => {
                // k = 1/T1b - 1/T1', with the scalar guard on T1b (gkm_filter.py:168-170).
                let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
                // The arterial decay factor is replaced by ZERO (not the quotient) for T1b <= 0
                // (gkm_filter.py:205, :218): `exp(-t/T1b) if t1b > 0 else 0`.
                let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
                if arriving {
                    // q_pasl_arriving, numerator and denominator computed separately so that
                    // t == dt cannot divide by zero (it is masked out anyway) (gkm_filter.py:173-185).
                    let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * t).exp());
                    let q = div0(num, kk * (t - dt));
                    2.0 * m0b * f * (t - dt) * k.alpha * decay * q
                } else {
                    let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * (dt + k.tau)).exp());
                    let q = div0(num, kk * k.tau);
                    2.0 * m0b * f * k.alpha * k.tau * decay * q
                }
            }
            LabelType::Casl | LabelType::Pcasl => {
                // Here the T1b test is `!= 0`, not `> 0` (gkm_filter.py:252, :265).
                let decay = if k.t1b != 0.0 { (-dt / k.t1b).exp() } else { 0.0 };
                if arriving {
                    let q = 1.0 - (-div0(t - dt, t1p)).exp();
                    2.0 * m0b * f * t1p * k.alpha * decay * q
                } else {
                    let q = 1.0 - (-div0(k.tau, t1p)).exp();
                    2.0 * m0b * f * t1p * k.alpha * decay * (-div0(t - k.tau - dt, t1p)).exp() * q
                }
            }
        }
    }

    const KS: [(f64, f64, f64, f64); 4] = [(60.0, 0.8, 1.33, 74.622), (20.0, 1.2, 0.83, 60.0), (0.0, 1000.0, 3.0, 90.0), (45.0, 0.5, 1.1, 1.0)];

    fn kinds() -> Vec<Kinetic> {
        vec![
            K_PCASL, K_PASL,
            Kinetic { label_type: LabelType::Casl, ..K_PCASL },
            Kinetic { lambda: 0.0, ..K_PCASL }, Kinetic { lambda: 0.0, ..K_PASL },
            Kinetic { t1b: 0.0, ..K_PCASL }, Kinetic { t1b: 0.0, ..K_PASL },
            Kinetic { t1b: -1.0, ..K_PCASL }, Kinetic { t1b: -1.0, ..K_PASL },
        ]
    }

    #[test]
    fn delta_m_keeps_its_bits_after_the_refactor() {
        let mut n = 0;
        for k in kinds() {
            for &(f, dt, t1t, m0) in &KS {
                for i in 0..=6000 {
                    let t = i as f64 * 1e-3;
                    let (a, b) = (delta_m(&k, f, dt, t1t, m0, t), delta_m_ref(&k, f, dt, t1t, m0, t));
                    assert_eq!(a.to_bits(), b.to_bits(), "{k:?} f {f} dt {dt} t {t}: {a:e} vs {b:e}");
                    n += 1;
                }
            }
        }
        // the zero-T1 tissue case and t on every mask edge
        for k in kinds() {
            for t in [0.8, 0.8 + k.tau, 0.0, -1.0] {
                assert_eq!(delta_m(&k, 60.0, 0.8, 0.0, 74.6, t).to_bits(), delta_m_ref(&k, 60.0, 0.8, 0.0, 74.6, t).to_bits());
            }
        }
        assert!(n > 200_000);
    }

    /// A deterministic pseudo-random partition of `[0, tau]` with 2..6 cuts, plus `extra` cuts.
    fn partition(seed: u64, tau: f64, extra: &[f64]) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let ncut = 2 + (next() * 5.0) as usize;
        let mut cuts: Vec<f64> = (0..ncut).map(|_| next() * tau).collect();
        cuts.extend(extra.iter().copied().filter(|c| *c > 0.0 && *c < tau));
        cuts.push(0.0);
        cuts.push(tau);
        cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cuts.dedup();
        cuts
    }

    /// `x` and its representable neighbours.
    fn around(x: f64) -> [f64; 3] {
        // wrapping: below 0.0 the neighbour is a NaN pattern, which the partition filters out
        [f64::from_bits(x.to_bits().wrapping_sub(1)), x, f64::from_bits(x.to_bits().wrapping_add(1))]
    }

    #[test]
    fn sub_boluses_sum_to_the_whole_bolus() {
        let mut worst = 0.0f64;
        for k in [K_PCASL, K_PASL, Kinetic { label_type: LabelType::Casl, ..K_PCASL }] {
            for &(f, dt, t1t, m0) in &KS[..2] {
                let peak = (0..=6000).map(|i| delta_m(&k, f, dt, t1t, m0, i as f64 * 1e-3).abs()).fold(0.0f64, f64::max);
                let mut times: Vec<f64> = (0..=600).map(|i| i as f64 * 1e-2).collect();
                for e in [dt, dt + k.tau] {
                    times.extend(around(e));
                }
                for (pi, &t) in times.iter().enumerate() {
                    // cuts at the representable neighbours of the delivery edges in the
                    // sub-bolus coordinate, as well as random ones
                    let edge = t - dt;
                    let extra: Vec<f64> = around(edge).iter().chain(around(edge - k.tau).iter()).copied().collect();
                    let cuts = partition(pi as u64 + 17, k.tau, &extra);
                    let sum: f64 = cuts.windows(2).map(|w| delta_m_sub(&k, f, dt, t1t, m0, t, w[0], w[1])).sum();
                    let whole = delta_m(&k, f, dt, t1t, m0, t);
                    if t <= dt {
                        assert_eq!(whole, 0.0);
                        assert_eq!(sum, 0.0, "{k:?} t {t}: not arrived must stay an exact zero");
                    }
                    let e = (sum - whole).abs() / peak;
                    worst = worst.max(e);
                    assert!(e <= 1e-12, "{k:?} f {f} t {t}: {sum:e} vs {whole:e} ({e:e} of peak)");
                }
            }
        }
        println!("sub-bolus partition identity: worst {worst:.2e} of peak");
        // the uncut sub-bolus is delta_m itself, bit for bit
        for t in [1.0, 2.0, 3.6, 5.0] {
            assert_eq!(delta_m_sub(&K_PCASL, 60.0, 0.8, 1.33, 74.6, t, 0.0, K_PCASL.tau).to_bits(),
                       delta_m(&K_PCASL, 60.0, 0.8, 1.33, 74.6, t).to_bits());
            assert_eq!(delta_m_sub(&K_PASL, 60.0, 0.8, 1.33, 74.6, t, 0.0, K_PASL.tau).to_bits(),
                       delta_m(&K_PASL, 60.0, 0.8, 1.33, 74.6, t).to_bits());
        }
    }

    #[test]
    fn intravascular_part_is_partition_invariant_and_bounded() {
        for k in [K_PCASL, K_PASL] {
            for tau_ex in [0.05, 0.5, 3.0] {
                let peak = (0..=600).map(|i| delta_m(&k, F, DT, T1T, M0, i as f64 * 1e-2)).fold(0.0f64, f64::max);
                for i in 0..=600 {
                    let t = i as f64 * 1e-2;
                    let dm = delta_m(&k, F, DT, T1T, M0, t);
                    let iv = delta_m_iv(&k, F, DT, T1T, M0, t, tau_ex);
                    assert!(iv.is_finite() && iv >= 0.0 && iv <= dm * (1.0 + 1e-12),
                            "{k:?} tau_ex {tau_ex} t {t}: iv {iv} dm {dm}");
                    let cuts = partition(i as u64 + 5, k.tau, &[]);
                    let sum: f64 = cuts.windows(2).map(|w| delta_m_iv_sub(&k, F, DT, T1T, M0, t, w[0], w[1], tau_ex)).sum();
                    assert!((sum - iv).abs() <= 1e-12 * peak, "{k:?} tau_ex {tau_ex} t {t}: {sum:e} vs {iv:e}");
                    if t <= DT {
                        assert_eq!(iv, 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn intravascular_part_limits_are_finite_in_both_branches() {
        for k in [K_PCASL, K_PASL] {
            // slow exchange: everything stays intravascular
            for t in [1.0, 1.4, 2.0, 3.6] {
                let dm = delta_m(&k, F, DT, T1T, M0, t);
                let iv = delta_m_iv(&k, F, DT, T1T, M0, t, 1e9);
                assert!((iv - dm).abs() <= 1e-8 * dm, "{k:?} t {t}: {iv} vs {dm}");
            }
            // fast exchange: just-arrived label has not exchanged yet; a second later it has
            let t = DT + 1e-7;
            let share = delta_m_iv(&k, F, DT, T1T, M0, t, 1e-6) / delta_m(&k, F, DT, T1T, M0, t);
            assert!((share - 0.95).abs() < 0.01, "{k:?}: share {share} at ATT + 1e-7");
            // both delivery branches at tau_ex = 1e-6 are finite (the shared PASL form gives NaN)
            for t in [DT + 0.3, DT + k.tau + 0.5, DT + 1.0] {
                let iv = delta_m_iv(&k, F, DT, T1T, M0, t, 1e-6);
                let dm = delta_m(&k, F, DT, T1T, M0, t);
                assert!(iv.is_finite(), "{k:?} t {t}: {iv}");
                if t >= DT + 1.0 {
                    assert!(iv / dm < 2e-6, "{k:?} t {t}: share {}", iv / dm);
                }
            }
        }
        // the PASL shared form really is the trap the stable form avoids
        let t1p = t1_prime(F, T1T, 0.9);
        let t1pp = 1.0 / (1.0 / t1p + 1e6);
        assert!(delta_m_with_t1p(&K_PASL, F, DT, t1pp, M0, DT + 1e-7).is_nan());
    }

    #[test]
    fn stable_pasl_form_equals_the_shared_form_where_that_is_finite() {
        let t1p = t1_prime(F, T1T, 0.9);
        for tau_ex in [0.2, 1.0, 10.0] {
            let t1pp = 1.0 / (1.0 / t1p + 1.0 / tau_ex);
            for i in 0..=400 {
                let t = i as f64 * 1e-2;
                let a = delta_m_iv(&K_PASL, F, DT, T1T, M0, t, tau_ex);
                let b = delta_m_with_t1p(&K_PASL, F, DT, t1pp, M0, t);
                assert!(a == b || (a - b).abs() <= 1e-12 * b.abs(), "t {t}: {a:e} vs {b:e}");
            }
        }
        // and delta_m_with_t1p at the GKM's own T1' is delta_m
        for t in [1.2, 2.0, 3.6] {
            assert_eq!(delta_m_with_t1p(&K_PCASL, F, DT, t1p, M0, t).to_bits(), delta_m(&K_PCASL, F, DT, T1T, M0, t).to_bits());
        }
    }

    #[test]
    fn pasl_intravascular_part_is_accurate_where_t1pp_nears_t1b() {
        // T1t = 3 puts the tau_ex that makes T1'' = T1b (kk = 0) in range: 1/tau_ex = 1/T1b - 1/T1'.
        // There the arrived branch's difference of two exponentials cancelled to noise, and the
        // computed part exceeded delta_m (Codex review: iv 0.01296 against delta_m 0.00567).
        let (t1t, m0, t) = (3.0, 1.0, 2.0);
        let t1p = t1_prime(F, t1t, 0.9);
        let tau_star = 1.0 / (1.0 / K_PASL.t1b - 1.0 / t1p);
        let dm = delta_m(&K_PASL, F, DT, t1t, m0, t);
        let iv = delta_m_iv(&K_PASL, F, DT, t1t, m0, t, 3.8223938223938227);
        assert!((iv - 0.004536224).abs() < 1e-8, "{iv} (delta_m {dm})");
        // the limit kk -> 0: q -> 1, so iv -> 2 M0b f alpha tau exp(-t/T1b)
        let limit = 2.0 * (m0 / 0.9) * (F / 6000.0) * K_PASL.alpha * K_PASL.tau * (-t / K_PASL.t1b).exp();
        let near = delta_m_iv(&K_PASL, F, DT, t1t, m0, t, tau_star * (1.0 + 1e-13));
        assert!((near - limit).abs() <= 1e-9 * limit, "{near} vs {limit}");
        // across the crossing, in both branches: bounded, and nondecreasing in tau_ex (slower
        // exchange keeps more label intravascular), which cancellation noise would break. A kk
        // of exactly zero takes the GKM's quotient guard (zero, which the spec keeps for every
        // P4 formula); it is skipped, and can happen at most once in the sweep.
        for t in [DT + 0.3, t, 3.0] {
            let dm = delta_m(&K_PASL, F, DT, t1t, m0, t);
            let (mut prev, mut zeros) = (0.0f64, 0);
            for j in -200..=200 {
                let tau_ex = tau_star * (1.0 + j as f64 * 1e-12);
                let iv = delta_m_iv(&K_PASL, F, DT, t1t, m0, t, tau_ex);
                assert!(iv >= 0.0 && iv <= dm * (1.0 + 1e-12), "t {t} tau_ex {tau_ex}: iv {iv} dm {dm}");
                if iv == 0.0 {
                    zeros += 1;
                    continue;
                }
                assert!(iv >= prev * (1.0 - 1e-12), "t {t} tau_ex {tau_ex}: {iv} after {prev}");
                prev = iv;
            }
            assert!(zeros <= 1, "t {t}: {zeros} guarded zeros");
        }
    }

    #[test]
    fn intravascular_guards_give_the_gkm_zeros() {
        for k in [Kinetic { lambda: 0.0, ..K_PCASL }, Kinetic { lambda: 0.0, ..K_PASL }, Kinetic { t1b: 0.0, ..K_PCASL },
                  Kinetic { t1b: 0.0, ..K_PASL }, Kinetic { t1b: -1.0, ..K_PASL }] {
            for t in [1.2, 2.0, 3.6] {
                assert_eq!(delta_m(&k, F, DT, T1T, M0, t), 0.0);
                assert_eq!(delta_m_iv(&k, F, DT, T1T, M0, t, 0.5), 0.0, "{k:?} t {t}");
            }
        }
        // zero flow and zero tissue T1: T1' is the guarded zero
        assert_eq!(delta_m_iv(&K_PCASL, 0.0, DT, 0.0, M0, 3.6, 0.5), 0.0);
    }

    #[test]
    fn arterial_term_follows_its_closed_forms() {
        let (abv, aatt) = (0.02, 0.5);
        for t in [0.5, 1.0, 2.29] {
            let (v, a) = arterial_dm(&K_PCASL, abv, aatt, M0, t);
            let want = 2.0 * 0.85 * (M0 / 0.9) * abv * (-aatt / 1.65f64).exp();
            assert!(close(v, want, 1e-14), "t {t}");
            assert_eq!(a, Some(t - aatt));
        }
        for t in [0.5, 0.9, 1.19] {
            let (v, a) = arterial_dm(&K_PASL, abv, aatt, M0, t);
            let want = 2.0 * 0.98 * (M0 / 0.9) * abv * (-t / 1.65f64).exp();
            assert!(close(v, want, 1e-14), "t {t}");
            assert_eq!(a, Some(t - aatt));
        }
        // the window: [aatt, aatt + tau)
        assert_eq!(arterial_dm(&K_PCASL, abv, aatt, M0, aatt - 1e-12), (0.0, None));
        assert_eq!(arterial_dm(&K_PCASL, abv, aatt, M0, aatt + 1.8), (0.0, None));
        assert!(arterial_dm(&K_PCASL, abv, aatt, M0, aatt).0 > 0.0);
        // guards
        assert_eq!(arterial_dm(&Kinetic { lambda: 0.0, ..K_PCASL }, abv, aatt, M0, 1.0).0, 0.0);
        assert_eq!(arterial_dm(&Kinetic { t1b: 0.0, ..K_PCASL }, abv, aatt, M0, 1.0).0, 0.0);
        assert_eq!(arterial_dm(&Kinetic { t1b: -1.0, ..K_PASL }, abv, aatt, M0, 1.0).0, 0.0);
        assert!(arterial_dm(&Kinetic { t1b: -1.0, ..K_PCASL }, abv, aatt, M0, 1.0).0 > 0.0,
                "(P)CASL tests != 0, as the GKM does");
    }

    // ---- P6 part B: arrival windows

    /// Windows partitioning [0, t) sum to the GKM at t (both labeling types; arrival before,
    /// during and after the windows; t inside and after the bolus).
    #[test]
    fn arrival_windows_partition_the_gkm() {
        for k in [K_PCASL, K_PASL] {
            for &(f, dt, t1t, m0) in &[(F, DT, T1T, M0), (20.0, 1.4, 0.83, 60.0), (60.0, 0.1, 1.33, 74.6)] {
                for t in [0.5, 0.9, 1.2, 1.9, 2.6, 3.4, 4.5] {
                    let whole = delta_m(&k, f, dt, t1t, m0, t);
                    for cuts in [vec![0.6], vec![0.3, 0.95, 1.7], vec![0.2, 0.4, 0.8, 1.1, 1.5, 2.1, 2.9]] {
                        let mut edges = vec![0.0];
                        edges.extend(cuts.iter().copied().filter(|&c| c < t));
                        edges.push(t);
                        let sum: f64 = edges.windows(2).map(|w| delta_m_arrival(&k, f, dt, t1t, m0, t, w[0], w[1])).sum();
                        assert!((sum - whole).abs() <= 1e-12 * whole.abs().max(1e-300) + 1e-15,
                                "{:?} t {t} dt {dt} cuts {cuts:?}: {sum} vs {whole}", k.label_type);
                    }
                }
            }
        }
    }

    /// Each window against the numerical quadrature of its integrand, the parcel arriving at u
    /// delivered at the GKM's rate and relaxing with T1' after it.
    #[test]
    fn arrival_windows_match_quadrature() {
        for k in [K_PCASL, K_PASL] {
            let (f, dt, t1t, m0) = (F, DT, T1T, M0);
            let (fs, m0b, t1p) = gkm_constants(&k, f, t1t, m0);
            let rate = |u: f64| -> f64 {
                if u < dt || u > dt + k.tau {
                    return 0.0;
                }
                let decay = match k.label_type {
                    LabelType::Pasl => (-u / k.t1b).exp(),
                    _ => (-dt / k.t1b).exp(),
                };
                2.0 * k.alpha * m0b * fs * decay
            };
            for (t, u1, u2) in [(2.0f64, 0.0f64, 1.0f64), (2.0, 0.9, 1.3), (3.0, 1.1, 2.6), (2.4, 1.6, 2.4), (1.0, 0.5, 0.9)] {
                let n = 200_000;
                let (lo, hi) = (u1.max(dt), u2.min(dt + k.tau).min(t));
                let quad = if hi > lo {
                    let h = (hi - lo) / n as f64;
                    (0..n).map(|i| {
                        let u = lo + (i as f64 + 0.5) * h;
                        rate(u) * (-(t - u) / t1p).exp()
                    }).sum::<f64>() * h
                } else {
                    0.0
                };
                let got = delta_m_arrival(&k, f, dt, t1t, m0, t, u1, u2);
                assert!((got - quad).abs() <= 1e-10 * quad.abs().max(1e-12) + 1e-14, "{:?} {t} [{u1}, {u2}): {got} vs {quad}", k.label_type);
            }
        }
    }

    /// The GKM's guards: PASL with q = 1/T1b - 1/T1' = 0 is zero (not the continuous limit), as
    /// delta_m is; no label before the arrival; T1b = 0 zeroes the PASL delivery.
    #[test]
    fn arrival_windows_keep_the_gkm_guards() {
        let (f, t1t, m0) = (F, T1T, M0);
        let (_, _, t1p) = gkm_constants(&K_PASL, f, t1t, m0);
        let k = Kinetic { t1b: t1p, ..K_PASL };
        assert_eq!(delta_m(&k, f, DT, t1t, m0, 1.2), 0.0);
        assert_eq!(delta_m_arrival(&k, f, DT, t1t, m0, 1.2, 0.0, 1.2), 0.0);
        assert_eq!(delta_m_arrival(&K_PCASL, f, DT, t1t, m0, 2.0, 0.0, DT), 0.0);
        assert_eq!(delta_m_arrival(&K_PCASL, f, DT, t1t, m0, 0.5, 0.0, 0.5), 0.0);
        let k0 = Kinetic { t1b: 0.0, ..K_PASL };
        assert_eq!(delta_m_arrival(&k0, f, DT, t1t, m0, 1.2, 0.0, 1.2), 0.0);
        assert!(delta_m_arrival(&K_PASL, f, DT, t1t, m0, 1.2, 0.0, 1.2) > 0.0);
    }

    /// The depleted read against a brute-force count: the label arriving in a small interval
    /// around u is depleted by every readout excited after u and before the one read. Includes a
    /// delayed slice whose first excitation follows the arrival.
    #[test]
    fn the_read_is_the_label_depleted_by_the_later_readouts() {
        for k in [K_PCASL, K_PASL] {
            for (dt, offset) in [(DT, 0.0), (1.05, 0.2), (0.3, 0.1)] {
                let t_n: Vec<f64> = (0..8).map(|n| 0.9 + 0.3 * n as f64).collect();
                let e: Vec<f64> = t_n.iter().map(|t| t + offset).collect();
                let flips: Vec<f64> = (0..8).map(|n| 25.0 + 5.0 * n as f64).collect();
                for n in 1..=8 {
                    let got = delta_m_read(&k, F, dt, T1T, M0, &e[..n], &flips[..n - 1]);
                    let t = e[n - 1];
                    let steps = 20_000;
                    let h = t / steps as f64;
                    let brute: f64 = (0..steps).map(|i| {
                        let (u0, u1) = (i as f64 * h, (i + 1) as f64 * h);
                        let u = 0.5 * (u0 + u1);
                        let w: f64 = (0..n - 1).filter(|&m| e[m] > u).map(|m| flips[m].to_radians().cos()).product();
                        w * delta_m_arrival(&k, F, dt, T1T, M0, t, u0, u1)
                    }).sum();
                    // the midpoint weight is exact except in the steps that straddle an excitation
                    assert!((got - brute).abs() <= 2e-4 * brute.abs().max(1e-9), "{:?} dt {dt} offset {offset} n {n}: {got} vs {brute}", k.label_type);
                    // with no earlier readouts (or zero flips) it is the undepleted delta_m
                    let zero = delta_m_read(&k, F, dt, T1T, M0, &e[..n], &vec![0.0; n - 1]);
                    let whole = delta_m(&k, F, dt, T1T, M0, t);
                    assert!((zero - whole).abs() <= 1e-12 * whole.abs().max(1e-300));
                }
            }
        }
    }

    // ---- P7 Task 1 ----

    fn rel_close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
    }

    /// Windows partitioning `[0, t)`: edges at 0, a few interior points, and `t`.
    fn edges(t: f64) -> Vec<f64> {
        let mut v = vec![0.0];
        v.extend([0.3, 0.7, 1.1, 1.6, 2.2, 2.9, 3.7].iter().copied().filter(|&x| x < t));
        v.push(t);
        v
    }

    /// `tau_ex` so long that `T1''` is `T1'` to the last bits: the intravascular window is the
    /// whole window.
    #[test]
    fn arrival_iv_without_exchange_is_the_arrival() {
        for k in [K_PCASL, K_PASL] {
            for t in [1.0, 1.6, 2.5, 3.6, 5.0] {
                for w in edges(t).windows(2) {
                    let whole = delta_m_arrival(&k, F, DT, T1T, M0, t, w[0], w[1]);
                    let iv = delta_m_arrival_iv(&k, F, DT, T1T, M0, t, w[0], w[1], 1e15);
                    assert!(rel_close(iv, whole, 1e-12), "{:?} t {t} {w:?}: {iv} vs {whole}", k.label_type);
                }
            }
        }
    }

    /// The intravascular windows partition `delta_m_iv`, for both labeling types, inside and after
    /// the bolus; for PASL also at a `tau_ex` where the naive exponent form is not finite.
    #[test]
    fn arrival_iv_windows_sum_to_delta_m_iv() {
        for k in [K_PCASL, K_PASL] {
            for tau_ex in [0.5, 1.5, 0.001] {
                for t in [1.0, 1.4, 2.5, 3.6, 5.0] {
                    let sum: f64 = edges(t).windows(2)
                        .map(|w| delta_m_arrival_iv(&k, F, DT, T1T, M0, t, w[0], w[1], tau_ex)).sum();
                    let want = delta_m_iv(&k, F, DT, T1T, M0, t, tau_ex);
                    assert!(sum.is_finite() && rel_close(sum, want, 1e-12),
                        "{:?} tau_ex {tau_ex} t {t}: {sum} vs {want}", k.label_type);
                }
            }
        }
        // the naive PASL form overflows there: kk = 1/T1b - 1/T1'' is about -1000
        let (_, _, t1pp) = t1pp(&K_PASL, F, T1T, M0, 0.001).unwrap();
        let kk = 1.0 / K_PASL.t1b - 1.0 / t1pp;
        let (t, u_lo, u_hi) = (3.6, 0.8, 1.5);
        let naive = (kk * t).exp() * ((-kk * u_lo).exp() - (-kk * u_hi).exp());
        assert!(!naive.is_finite(), "the naive form is finite here ({naive}): the test does not exercise the stable form");
    }

    /// No window's intravascular part is negative or exceeds its whole.
    #[test]
    fn arrival_iv_is_clamped_to_its_window() {
        for k in [K_PCASL, K_PASL] {
            for tau_ex in [0.01, 0.3, 2.0] {
                for t in [1.0, 2.0, 3.6] {
                    for w in edges(t).windows(2) {
                        let whole = delta_m_arrival(&k, F, DT, T1T, M0, t, w[0], w[1]);
                        let iv = delta_m_arrival_iv(&k, F, DT, T1T, M0, t, w[0], w[1], tau_ex);
                        assert!(iv >= 0.0 && iv <= whole * (1.0 + 1e-14), "{:?}: {iv} of {whole}", k.label_type);
                    }
                }
            }
        }
    }

    /// The sub-boli of a partition of the bolus sum, window by window, to the whole window; and
    /// over both partitions to `delta_m`.
    #[test]
    fn arrival_sub_partitions_sum_to_the_whole() {
        for k in [K_PCASL, K_PASL] {
            let cuts = [0.0, 0.2 * k.tau, 0.55 * k.tau, k.tau];
            for t in [1.0, 1.6, 2.5, 3.6, 5.0] {
                let mut total = 0.0;
                for w in edges(t).windows(2) {
                    let whole = delta_m_arrival(&k, F, DT, T1T, M0, t, w[0], w[1]);
                    let parts: f64 = cuts.windows(2)
                        .map(|c| delta_m_arrival_sub(&k, F, DT, T1T, M0, t, w[0], w[1], c[0], c[1])).sum();
                    assert!((parts - whole).abs() <= 1e-12 * whole.abs().max(1e-12), "{:?} t {t} {w:?}: {parts} vs {whole}", k.label_type);
                    let iv_parts: f64 = cuts.windows(2)
                        .map(|c| delta_m_arrival_iv_sub(&k, F, DT, T1T, M0, t, w[0], w[1], c[0], c[1], 0.7)).sum();
                    let iv = delta_m_arrival_iv(&k, F, DT, T1T, M0, t, w[0], w[1], 0.7);
                    assert!((iv_parts - iv).abs() <= 1e-12 * iv.abs().max(1e-12), "iv {:?} t {t}: {iv_parts} vs {iv}", k.label_type);
                    total += parts;
                }
                let want = delta_m(&k, F, DT, T1T, M0, t);
                assert!((total - want).abs() <= 1e-12 * want.abs().max(1e-12), "{:?} t {t}: {total} vs {want}", k.label_type);
            }
        }
    }

    fn train(n: usize, start: f64, step: f64) -> (Vec<f64>, Vec<f64>) {
        ((0..n).map(|m| start + step * m as f64).collect(), (0..n).map(|m| 20.0 + 3.0 * (m % 7) as f64).collect())
    }

    /// With one full sub-bolus, no exchange and no shift, the parts read is P6's read.
    #[test]
    fn read_parts_reduces_to_delta_m_read() {
        for k in [K_PCASL, K_PASL] {
            let (e, flips) = train(10, 0.6, 0.3);
            for n in 1..=10 {
                let p = delta_m_read_parts(&k, F, DT, T1T, M0, &e[..n], &flips[..n - 1], &[(0.0, k.tau, 1.0)], None, 0.0);
                let want = delta_m_read(&k, F, DT, T1T, M0, &e[..n], &flips[..n - 1]);
                assert_eq!(p.ev, 0.0);
                assert!(rel_close(p.iv, want, 1e-12), "{:?} n {n}: {} vs {want}", k.label_type, p.iv);
            }
        }
    }

    /// Slab entry by hand: excitations at 1.0 and 1.2 s, `delta = 0.5`. Every parcel arriving
    /// before the read at 1.2 entered the slab before 1.0 + 0.5, so the first excitation depleted
    /// all of it: the read is `cos(a0) delta_m(1.2)`. With `delta = 0` the window `[1.0, 1.2)` is
    /// undepleted.
    #[test]
    fn slab_entry_depletes_label_not_yet_arrived() {
        let k = K_PCASL;
        let e = [1.0, 1.2];
        let a0: f64 = 40.0;
        let full = delta_m(&k, F, 0.3, T1T, M0, 1.2);
        let shifted = delta_m_read_parts(&k, F, 0.3, T1T, M0, &e, &[a0], &[(0.0, k.tau, 1.0)], None, 0.5);
        assert!(rel_close(shifted.iv, a0.to_radians().cos() * full, 1e-12), "{} vs {}", shifted.iv, a0.to_radians().cos() * full);
        let arrival = delta_m_read_parts(&k, F, 0.3, T1T, M0, &e, &[a0], &[(0.0, k.tau, 1.0)], None, 0.0);
        let late = delta_m_arrival(&k, F, 0.3, T1T, M0, 1.2, 1.0, 1.2);
        assert!(rel_close(arrival.iv, a0.to_radians().cos() * (full - late) + late, 1e-12));
        assert!(arrival.iv > shifted.iv);
    }

    /// Every read of a train in O(n) equals the direct sum at every excitation: both labeling
    /// types, with and without exchange, with sub-boli and factors, at no shift and two shifts;
    /// and a long train late on the clock stays finite.
    #[test]
    fn read_all_equals_the_direct_reads() {
        for k in [K_PCASL, K_PASL] {
            let subs_cut = [(0.0, 0.3 * k.tau, -1.0), (0.3 * k.tau, k.tau, 0.8)];
            let subs_one = [(0.0, k.tau, 1.0)];
            for (e, flips) in [train(12, 0.6, 0.3), train(48, 1.0, 0.04), train(64, 30.0, 0.05)] {
                let dt = if e[0] > 10.0 { 29.5 } else { DT };
                for subs in [&subs_one[..], &subs_cut[..]] {
                    for tau_ex in [None, Some(0.6), Some(0.001)] {
                        for delta in [0.0, 0.17, 0.8] {
                            let all = delta_m_read_all(&k, F, dt, T1T, M0, &e, &flips, subs, tau_ex, delta);
                            for n in 1..=e.len() {
                                let p = delta_m_read_parts(&k, F, dt, T1T, M0, &e[..n], &flips[..n - 1], subs, tau_ex, delta);
                                let peak = p.iv.abs().max(p.ev.abs()).max(1e-12);
                                assert!(all[n - 1].iv.is_finite() && all[n - 1].ev.is_finite());
                                assert!((all[n - 1].iv - p.iv).abs() <= 1e-12 * peak && (all[n - 1].ev - p.ev).abs() <= 1e-12 * peak,
                                    "{:?} n {n} tau_ex {tau_ex:?} delta {delta}: {:?} vs {p:?}", k.label_type, all[n - 1]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The cost of every read of a train is linear in its length: run with `--ignored
    /// --nocapture` to record it (P7 plan, Measurements).
    #[test]
    #[ignore]
    fn read_all_cost_is_linear() {
        let subs = [(0.0, K_PCASL.tau, 1.0)];
        let mut per = Vec::new();
        for n in [48, 96] {
            let (e, flips) = train(n, 1.0, 0.04);
            let start = std::time::Instant::now();
            let mut acc = 0.0;
            for v in 0..100_000 {
                let f = 40.0 + (v % 50) as f64;
                acc += delta_m_read_all(&K_PCASL, f, DT, T1T, M0, &e, &flips, &subs, Some(0.6), 0.3)[n - 1].iv;
            }
            let s = start.elapsed().as_secs_f64();
            println!("1e5 voxels x {n} excitations: {s:.2} s ({acc:.3e})");
            per.push(s);
        }
        assert!(per[1] < 3.0 * per[0], "doubling the train more than tripled the time: {per:?}");
    }

    /// The arterial read: fresh at `delta_a = 0`; the cosines of the excitations inside
    /// `[e_n - delta_a, e_n)`, the left end included and `e_n` itself not.
    #[test]
    fn arterial_read_depletes_from_slab_entry() {
        let (abv, aatt) = (0.02, 0.5);
        let e_n = 1.0;
        let (fresh, _) = arterial_dm(&K_PCASL, abv, aatt, M0, e_n);
        assert!(fresh > 0.0);
        let earlier = [(0.55, 30.0), (0.7, 40.0), (0.9, 20.0), (1.0, 50.0)];
        assert_eq!(arterial_read(&K_PCASL, abv, aatt, M0, e_n, &earlier, 0.0, 1.0), fresh);
        let c = |a: f64| a.to_radians().cos();
        let two = arterial_read(&K_PCASL, abv, aatt, M0, e_n, &earlier, 0.3, 1.0);
        assert!(rel_close(two, fresh * c(40.0) * c(20.0), 1e-15), "{two}");
        let edge = arterial_read(&K_PCASL, abv, aatt, M0, e_n, &earlier, 0.45, 1.0);
        assert!(rel_close(edge, fresh * c(30.0) * c(40.0) * c(20.0), 1e-15), "{edge}");
        assert_eq!(arterial_read(&K_PCASL, abv, aatt, M0, e_n, &earlier, 0.3, 0.25), two * 0.25);
        // outside the arterial window there is nothing to read
        assert_eq!(arterial_read(&K_PCASL, abv, aatt, M0, 0.4, &earlier, 0.3, 1.0), 0.0);
    }

    // ---- P7 Task 2: the parcel reference ----

    use super::parcel_ref::{Case as PCase, Region as PRegion};

    fn case(k: Kinetic, att: f64, t: f64) -> PCase {
        PCase {
            k, f: F, att, t1t: T1T, m0: M0, t, excitations: vec![], entry_lead: 0.0, pulses: vec![], epsilon: 0.0,
            region: PRegion::Global, tau_ex: None, span: (0.0, k.tau),
        }
    }

    /// The reference in its trivial limits: no excitations and no pulses is `delta_m`; without
    /// exchange the intravascular part is all of it; P4's worked slab-confined factor for
    /// `asl002` (0.0821045); and it has converged.
    #[test]
    fn the_parcel_reference_reproduces_the_closed_forms() {
        for k in [K_PCASL, K_PASL] {
            for t in [1.0, 1.4, 2.5, 3.6] {
                let c = case(k, DT, t);
                let (iv, total) = c.read(4);
                let want = delta_m(&k, F, DT, T1T, M0, t);
                assert!(rel_close(total, want, 1e-12), "{:?} t {t}: {total} vs {want}", k.label_type);
                assert_eq!(iv, total);
                let ex = PCase { tau_ex: Some(0.6), ..c.clone() }.read(4).0;
                let want_iv = delta_m_iv(&k, F, DT, T1T, M0, t, 0.6);
                assert!(rel_close(ex, want_iv, 1e-12), "{:?} t {t}: iv {ex} vs {want_iv}", k.label_type);
            }
        }
        let k = Kinetic { tau: 1.8, ..K_PCASL };
        let plain = case(k, 0.8, 3.6);
        let supp = PCase { pulses: vec![2.05, 3.276], epsilon: 1.0, region: PRegion::Arrival, ..plain.clone() };
        let ratio = supp.read(4).1 / plain.read(4).1;
        assert!((ratio - 0.082_104_5).abs() < 1e-6, "{ratio}");
        // two resolutions agree
        let mut c = case(K_PASL, 0.6, 2.4);
        c.excitations = vec![(1.0, 30.0), (1.3, 30.0), (1.6, 30.0), (1.9, 30.0)];
        c.entry_lead = 0.25;
        c.tau_ex = Some(0.4);
        let (a, b) = (c.read(4), c.read(8));
        assert!((a.0 - b.0).abs() < 1e-7 * a.0.abs() && (a.1 - b.1).abs() < 1e-7 * a.1.abs(), "{a:?} vs {b:?}");
    }

    /// Task 1's closed forms, with P4's sub-bolus factors, against the parcel reference: both
    /// labeling types, with exchange, bolus-position `"slab"`, `"global"` and `"arrival"`, with and
    /// without a slab-entry shift, arrival before, during and after the readouts.
    #[test]
    fn depleted_reads_match_the_parcel_reference() {
        use crate::bolus::{entry_offset, subbolus_factors, Region};
        let e: Vec<f64> = (0..8).map(|n| 1.0 + 0.3 * n as f64).collect();
        let flips: Vec<f64> = (0..8).map(|n| 25.0 + 4.0 * n as f64).collect();
        let pulses = [0.2, 0.9];
        let mut checked = 0;
        for k in [K_PCASL, K_PASL] {
            for att in [0.3, 1.0, 1.6] {
                for (region, pregion) in [(Region::Slab(0.3), PRegion::Slab(0.3)), (Region::Global, PRegion::Global),
                                          (Region::Arrival, PRegion::Arrival)] {
                    let subs = subbolus_factors(&pulses, 0.93, k.tau, entry_offset(k.label_type, region, att));
                    for tau_ex in [None, Some(0.6)] {
                        for lead in [0.0, 0.25] {
                            let all = delta_m_read_all(&k, F, att, T1T, M0, &e, &flips, &subs, tau_ex, lead);
                            for n in 1..=e.len() {
                                let p = delta_m_read_parts(&k, F, att, T1T, M0, &e[..n], &flips[..n - 1], &subs, tau_ex, lead);
                                let c = PCase {
                                    excitations: e[..n - 1].iter().copied().zip(flips.iter().copied()).collect(),
                                    entry_lead: lead, pulses: pulses.to_vec(), epsilon: 0.93, region: pregion, tau_ex,
                                    ..case(k, att, e[n - 1])
                                };
                                let (iv, total) = c.read(4);
                                let scale = total.abs().max(1e-9);
                                assert!((p.total() - total).abs() <= 1e-6 * scale && (p.iv - iv).abs() <= 1e-6 * scale,
                                    "{:?} att {att} {region:?} tau_ex {tau_ex:?} lead {lead} n {n}: {p:?} vs iv {iv} total {total}",
                                    k.label_type);
                                assert!((all[n - 1].total() - total).abs() <= 1e-6 * scale);
                                checked += usize::from(total.abs() > 1e-6);
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 300, "too few nonzero reads checked: {checked}");
    }

    /// The arterial read, with P4's parcel factor, against the parcel reference: fresh and
    /// depleted from slab entry, under each region.
    #[test]
    fn arterial_reads_match_the_parcel_reference() {
        use crate::bolus::{arterial_factor, entry_offset, Region};
        let excitations: Vec<(f64, f64)> = (0..8).map(|n| (0.6 + 0.15 * n as f64, 30.0 + 5.0 * n as f64)).collect();
        let pulses = [0.2, 0.5];
        for k in [K_PCASL, K_PASL] {
            let aatt = 0.4;
            for (region, pregion) in [(Region::Slab(0.1), PRegion::Slab(0.1)), (Region::Global, PRegion::Global)] {
                for lead in [0.0, 0.3] {
                    for n in 0..excitations.len() {
                        let t = excitations[n].0;
                        let fac = arterial_factor(&pulses, 0.9, t - aatt, entry_offset(k.label_type, region, aatt));
                        let got = arterial_read(&k, 0.02, aatt, M0, t, &excitations[..n], lead, fac);
                        let c = PCase { excitations: excitations[..n].to_vec(), pulses: pulses.to_vec(), epsilon: 0.9,
                                       region: pregion, ..case(k, DT, t) };
                        let want = c.arterial(0.02, aatt, lead);
                        assert!(rel_close(got, want, 1e-12), "{:?} {region:?} lead {lead} n {n}: {got} vs {want}", k.label_type);
                    }
                }
            }
        }
    }
}
