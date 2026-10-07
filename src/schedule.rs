//! The schedule (P6 addendum, "The one idea underneath"): what the scanner does, in three index
//! levels, separate from what is written.
//!
//! - A **preparation** is one labeling (or none, for an m0scan row): its start on the series clock,
//!   its labeling window, the suppression pulse set and crushing VENC that apply to it.
//! - A **raw volume** is one volume of the acquisition call. In 2D it holds one preparation; a
//!   segmented 3D volume holds `NumberShots` of them (P5 part D); under Look-Locker several raw
//!   volumes (the readouts) share one preparation.
//! - An **output volume** is what the dataset lists: a raw volume, or a decoded one (Hadamard).
//!
//! `Protocol.rows` stays the input/output rows (what `aslcontext.tsv` lists). Every protocol
//! without a P6 feature has the [`Schedule::identity`] layout, and the series runs today's code
//! for it (the legacy dispatch, `series.rs`); the identity schedule exists so the P6 path and its
//! tests can state that layout, not so the legacy path can read it.

use crate::kinetic::LabelType;
use crate::protocol::Protocol;
use crate::rows::{Row, RowKind};

/// One labeling (P6: "preparation").
#[derive(Debug, Clone, PartialEq)]
pub struct Preparation {
    /// The raw volume this preparation is read in (the first, under Look-Locker).
    pub raw: usize,
    /// Its shot within that raw volume (0 in 2D).
    pub shot: usize,
    /// Start of the repetition on the series clock (s): the labeling starts here.
    pub start_s: f64,
    /// The labeling window on the series clock (s): `[start, start + tau]` for (P)CASL, the
    /// instant `[start, start]` for PASL and m0scan.
    pub labeling_window: [f64; 2],
    /// Index into `SuppressionSpec::per_row` of the pulse set this labeling runs (when
    /// suppression is on).
    pub suppression: usize,
    /// The crushing VENC (cm/s) of this labeling, when crushing is on.
    pub venc: Option<f64>,
}

/// One volume of the acquisition call.
#[derive(Debug, Clone, PartialEq)]
pub struct RawVolume {
    /// Its first preparation.
    pub prep: usize,
    /// How many preparations it holds (`NumberShots` in segmented 3D, else 1).
    pub n_preps: usize,
    /// Its readout index within the preparation (0 except under Look-Locker).
    pub readout: usize,
    /// The Hadamard or Look-Locker cycle it belongs to.
    pub cycle: Option<usize>,
    /// Hadamard: its row of the encoding (`crate::hadamard::encoding`), which every one of its
    /// preparations repeats.
    pub encoding_row: Option<usize>,
    /// The crushing VENC (cm/s) of this readout, when it is its own (P7 addendum, part A: under
    /// Look-Locker each readout carries its bipolar gradients, so `VascularCrushingVENC` is per
    /// readout). `None` elsewhere: the readout takes its preparation's.
    pub venc: Option<f64>,
}

impl RawVolume {
    /// The VENC this raw volume is read with: its own under Look-Locker, else its first
    /// preparation's.
    pub fn venc_with(&self, preps: &[Preparation]) -> Option<f64> {
        self.venc.or(preps[self.prep].venc)
    }
}

/// One output volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// A raw volume, written as acquired.
    Raw(usize),
    /// Sub-bolus `subbolus` of Hadamard cycle `cycle`, decoded; at Look-Locker readout `readout`
    /// (P7 part B; 0 without Look-Locker).
    Decoded { cycle: usize, subbolus: usize, readout: usize },
}

/// A Hadamard or Look-Locker cycle: a run of raw volumes that belong together.
#[derive(Debug, Clone, PartialEq)]
pub struct Cycle {
    /// The raw volumes, `raws[start..end]`.
    pub raws: std::ops::Range<usize>,
    /// The input/output rows it produces (Hadamard: its decoded rows, in sub-bolus order).
    pub rows: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    /// One row per raw volume, with its own timing and kind.
    pub raw_rows: Vec<Row>,
    pub preps: Vec<Preparation>,
    pub raws: Vec<RawVolume>,
    pub outputs: Vec<Output>,
    pub cycles: Vec<Cycle>,
}

impl Schedule {
    /// The protocol's schedule: Hadamard's under `[hadamard]`, else the identity.
    pub fn new(p: &Protocol) -> Schedule {
        match (&p.hadamard, &p.look_locker) {
            (Some(h), _) => Schedule::hadamard(p, h),
            (None, Some(ll)) => Schedule::look_locker(p, ll),
            (None, None) => Schedule::identity(p),
        }
    }

    /// Look-Locker (P6 addendum, part B): one preparation per cycle, its readouts the cycle's raw
    /// volumes (each its own output, in the input order); an m0scan row is its own one-readout
    /// cycle; the clock advances one repetition per cycle. In segmented 3D (P7 part C) a cycle is
    /// `NumberShots` preparations, one per shot, every readout's raw volume reading each of them
    /// (shot `s` one repetition after shot `s - 1`): the clock advances `NumberShots` repetitions.
    pub fn look_locker(p: &Protocol, ll: &crate::protocol::LookLockerSpec) -> Schedule {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
        let mut sched = Schedule { raw_rows: Vec::new(), preps: Vec::new(), raws: Vec::new(), outputs: Vec::new(), cycles: Vec::new() };
        let mut clock = 0.0;
        for (c, cy) in ll.cycles.iter().enumerate() {
            let first = &p.rows[cy.rows[0]];
            let start = clock;
            let end = match (cy.m0scan, p.label_type) {
                (true, _) | (_, LabelType::Pasl) => start,
                _ => start + first.tau,
            };
            let prep = sched.preps.len();
            let r0 = sched.raws.len();
            for s in 0..shots {
                let d = s as f64 * first.tr;
                sched.preps.push(Preparation {
                    raw: r0, shot: s, start_s: start + d, labeling_window: [start + d, end + d], suppression: cy.rows[0],
                    venc: p.crushing.as_ref().map(|cr| cr.venc[cy.rows[0]]),
                });
            }
            for (n, &v) in cy.rows.iter().enumerate() {
                sched.raws.push(RawVolume {
                    prep, n_preps: shots, readout: n, cycle: Some(c), encoding_row: None,
                    venc: p.crushing.as_ref().map(|cr| cr.venc[v]),
                });
                sched.raw_rows.push(p.rows[v].clone());
                sched.outputs.push(Output::Raw(r0 + n));
            }
            sched.cycles.push(Cycle { raws: r0..r0 + cy.rows.len(), rows: cy.rows.clone() });
            clock += shots as f64 * first.tr;
        }
        sched
    }

    /// Hadamard (P6 addendum, part A, "The schedule"): an `m0scan` row is one raw volume, as today;
    /// a cycle is `H` raw volumes in encoding-row order, each a `label` row at
    /// `t = tau_tot + PLD`, `tau = tau_tot` (its blood is the encoded sum of sub-boli) and each
    /// `NumberShots` preparations (3D) repeating its row; the clock advances `NumberShots x tr`
    /// per raw volume. The outputs follow the input rows: an `m0scan` row is its raw volume, a
    /// `deltam` row its cycle's decoded sub-bolus. Pulse sets and VENC are the cycle's (the
    /// protocol checked they are one per cycle), taken from its first row.
    ///
    /// Under Look-Locker (P7 addendum, part B) each encoding row's preparations are read `M` times:
    /// `M` raw volumes share them (readout `n` at `t = tau_tot + PLD_n`, each with its own VENC),
    /// so a cycle is `H M` raw volumes, encoding-row-major; its decoded outputs are readout-major,
    /// as the rows are.
    pub fn hadamard(p: &Protocol, h: &crate::protocol::HadamardSpec) -> Schedule {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
        let mut sched = Schedule { raw_rows: Vec::new(), preps: Vec::new(), raws: Vec::new(), outputs: Vec::new(), cycles: Vec::new() };
        let mut clock = 0.0;
        let m = h.readouts;
        // one encoding row's preparations and its raw volumes: `readouts` rows read after the same
        // preparations (one row without Look-Locker), each with its source row's VENC under
        // Look-Locker
        let mut push_raw = |sched: &mut Schedule, readouts: &[(Row, usize)], source_row: usize, labeled: bool,
                            cycle: Option<usize>, enc: Option<usize>| {
            let r = sched.raws.len();
            let prep = sched.preps.len();
            for (n, (row, vrow)) in readouts.iter().enumerate() {
                let venc = if m > 1 { p.crushing.as_ref().map(|c| c.venc[*vrow]) } else { None };
                sched.raws.push(RawVolume { prep, n_preps: shots, readout: n, cycle, encoding_row: enc, venc });
                sched.raw_rows.push(row.clone());
            }
            let row = &readouts[0].0;
            for s in 0..shots {
                let start = clock + s as f64 * row.tr;
                sched.preps.push(Preparation {
                    raw: r,
                    shot: s,
                    start_s: start,
                    labeling_window: [start, if labeled { start + row.tau } else { start }],
                    suppression: source_row,
                    venc: p.crushing.as_ref().map(|c| c.venc[source_row]),
                });
            }
            clock += shots as f64 * row.tr;
            r
        };
        let mut v = 0;
        while v < p.rows.len() {
            if let Some(c) = h.cycles.iter().position(|cy| cy.rows[0] == v) {
                let cy = &h.cycles[c];
                let tr = p.rows[v].tr;
                let first = sched.raws.len();
                let readouts: Vec<(Row, usize)> = (0..m)
                    .map(|n| (Row { kind: RowKind::Label, t: h.tau_tot + h.plds[n], tau: h.tau_tot, tr }, h.row(cy, 0, n)))
                    .collect();
                for e in 0..h.order {
                    push_raw(&mut sched, &readouts, v, true, Some(c), Some(e));
                }
                sched.cycles.push(Cycle { raws: first..first + h.order * m, rows: cy.rows.clone() });
                for n in 0..m {
                    for j in 0..h.order - 1 {
                        sched.outputs.push(Output::Decoded { cycle: c, subbolus: j, readout: n });
                    }
                }
                v += cy.rows.len();
            } else {
                // an m0scan row (the protocol allows nothing else outside a cycle)
                let r = push_raw(&mut sched, &[(p.rows[v].clone(), v)], v, false, None, None);
                sched.outputs.push(Output::Raw(r));
                v += 1;
            }
        }
        sched
    }

    /// Today's layout: one raw volume per row, `NumberShots` preparations per raw volume in
    /// segmented 3D (one in 2D), each on the series clock of `Protocol::row_start` (shot `s` of
    /// row `v` starts at `row_start[v] + s * tr`), one output per raw volume.
    pub fn identity(p: &Protocol) -> Schedule {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
        let mut preps = Vec::with_capacity(p.rows.len() * shots);
        let mut raws = Vec::with_capacity(p.rows.len());
        for (v, row) in p.rows.iter().enumerate() {
            raws.push(RawVolume { prep: preps.len(), n_preps: shots, readout: 0, cycle: None, encoding_row: None, venc: None });
            for s in 0..shots {
                let start = p.row_start[v] + s as f64 * row.tr;
                let end = match p.label_type {
                    LabelType::Pasl => start,
                    _ => start + row.tau,
                };
                preps.push(Preparation {
                    raw: v,
                    shot: s,
                    start_s: start,
                    labeling_window: [start, end],
                    suppression: v,
                    venc: p.crushing.as_ref().map(|c| c.venc[v]),
                });
            }
        }
        Schedule {
            raw_rows: p.rows.clone(),
            preps,
            outputs: (0..p.rows.len()).map(Output::Raw).collect(),
            raws,
            cycles: Vec::new(),
        }
    }

    /// The preparations of raw volume `r`.
    pub fn preps_of(&self, r: usize) -> &[Preparation] {
        let rv = &self.raws[r];
        &self.preps[rv.prep..rv.prep + rv.n_preps]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{parse, Overlay};
    use serde_json::{json, Value};

    /// H8, two cycles of equal sub-boli (0.25 s, PLD 0.2 s), an m0scan row first at TR 6 s.
    fn h8(three_d: bool) -> Protocol {
        let mut ld = vec![0.0];
        let mut pld = vec![0.0];
        for _ in 0..2 {
            for j in 0..7 {
                ld.push(0.25);
                pld.push(0.2 + 0.25 * (6 - j) as f64);
            }
        }
        let mut tr = vec![6.0];
        tr.extend(vec![4.0; 14]);
        let mut s: Value = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": "Included", "RepetitionTimePreparation": tr,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [3.5, 3.5, 5],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.016
        });
        if three_d {
            let o = s.as_object_mut().unwrap();
            o.remove("SliceTiming");
            o.remove("TotalReadoutTime");
            s["MRAcquisitionType"] = json!("3D");
            s["PulseSequenceType"] = json!("3Dgrase");
            s["EffectiveEchoSpacing"] = json!(0.0003);
            s["NumberShots"] = json!(4);
            s["FlipAngle"] = json!(150);
        }
        let ov: Overlay = toml::from_str("[hadamard]\norder = 8\n").unwrap();
        let ctx = format!("volume_type\nm0scan\n{}", "deltam\n".repeat(14));
        parse(&s, &ctx, Some(&ov), None).unwrap()
    }

    #[test]
    fn the_hadamard_schedule() {
        for (three_d, shots) in [(false, 1usize), (true, 4)] {
            let p = h8(three_d);
            let s = Schedule::new(&p);
            // an m0scan raw volume, then two cycles of eight
            assert_eq!((s.raws.len(), s.raw_rows.len(), s.preps.len(), s.outputs.len()), (17, 17, 17 * shots, 15));
            assert_eq!(s.cycles.iter().map(|c| c.raws.clone()).collect::<Vec<_>>(), vec![1..9, 9..17]);
            assert_eq!(s.cycles[1].rows, (8..15).collect::<Vec<_>>());
            assert_eq!(s.outputs[0], Output::Raw(0));
            assert_eq!(s.outputs[1], Output::Decoded { cycle: 0, subbolus: 0, readout: 0 });
            assert_eq!(s.outputs[14], Output::Decoded { cycle: 1, subbolus: 6, readout: 0 });
            assert_eq!((s.raws[0].cycle, s.raws[0].encoding_row, s.raw_rows[0].kind), (None, None, RowKind::M0scan));
            let mut clock = 0.0;
            for (r, raw) in s.raws.iter().enumerate() {
                let row = &s.raw_rows[r];
                if r > 0 {
                    // every encoded raw volume is a label row read at tau_tot + PLD
                    assert_eq!((raw.cycle, raw.encoding_row), (Some((r - 1) / 8), Some((r - 1) % 8)));
                    assert_eq!((row.kind, row.tau, row.tr), (RowKind::Label, 1.75, 4.0));
                    assert!((row.t - 1.95).abs() < 1e-12);
                    // the readout at the raw t: the first sub-bolus's own PLD + tau
                    let first = s.cycles[raw.cycle.unwrap()].rows[0];
                    assert!((row.t - p.rows[first].t).abs() < 1e-12);
                }
                assert_eq!(raw.n_preps, shots);
                for (k, prep) in s.preps_of(r).iter().enumerate() {
                    assert_eq!((prep.raw, prep.shot), (r, k));
                    assert_eq!(prep.start_s, clock + k as f64 * row.tr);
                    let end = if r == 0 { prep.start_s } else { prep.start_s + 1.75 };
                    assert_eq!(prep.labeling_window, [prep.start_s, end]);
                    assert_eq!(prep.suppression, if r == 0 { 0 } else { 1 + 7 * ((r - 1) / 8) });
                }
                clock += shots as f64 * row.tr;
            }
            // the m0scan row at its own 6 s
            assert_eq!(s.preps_of(1)[0].start_s, shots as f64 * 6.0);
        }
    }
}
