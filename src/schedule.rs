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
use crate::rows::Row;

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
}

/// One output volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// A raw volume, written as acquired.
    Raw(usize),
    /// Sub-bolus `subbolus` of Hadamard cycle `cycle`, decoded.
    Decoded { cycle: usize, subbolus: usize },
}

/// A Hadamard or Look-Locker cycle: a run of raw volumes that belong together.
#[derive(Debug, Clone, PartialEq)]
pub struct Cycle {
    /// The raw volumes, `raws[start..end]`.
    pub raws: std::ops::Range<usize>,
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
    /// Today's layout: one raw volume per row, `NumberShots` preparations per raw volume in
    /// segmented 3D (one in 2D), each on the series clock of `Protocol::row_start` (shot `s` of
    /// row `v` starts at `row_start[v] + s * tr`), one output per raw volume.
    pub fn identity(p: &Protocol) -> Schedule {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
        let mut preps = Vec::with_capacity(p.rows.len() * shots);
        let mut raws = Vec::with_capacity(p.rows.len());
        for (v, row) in p.rows.iter().enumerate() {
            raws.push(RawVolume { prep: preps.len(), n_preps: shots, readout: 0, cycle: None });
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
