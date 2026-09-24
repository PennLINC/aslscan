//! `aslscan`: a BIDS ASL protocol and a BIDS-derivatives phantom in, a BIDS ASL dataset out.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use aslscan::bids::{write_dataset, Names};
use aslscan::phantom::{self, T2Mode};
use aslscan::protocol;
use aslscan::series;
use mrsim_acq::phase::PhaseModel;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Auto,
    Class,
    Voxel,
}

#[derive(Parser, Debug)]
#[command(name = "aslscan", version, about = "Simulate a BIDS ASL series from a BIDS-derivatives phantom")]
struct Cli {
    /// The BIDS ASL sidecar (`*_asl.json`) describing the protocol.
    #[arg(long, value_name = "JSON")]
    asl_json: PathBuf,
    /// The `*_aslcontext.tsv` giving the volume order.
    #[arg(long, value_name = "TSV")]
    aslcontext: PathBuf,
    /// The phantom directory (perfusion, att, T1map, T2map, T2starmap, M0map, dseg, ...).
    #[arg(long, value_name = "DIR")]
    phantom: PathBuf,
    /// Optional TOML overlay: kinetic constants, signal, acquisition knobs, M0 repetition time.
    #[arg(long, value_name = "TOML")]
    overlay: Option<PathBuf>,
    /// How transverse relaxation is represented: one compartment per label with uniform T2/T2'
    /// (class), one tissue compartment with per-voxel maps (voxel), or class when the phantom's
    /// maps are constant within each label (auto).
    #[arg(long, value_enum, default_value_t = Mode::Auto)]
    t2_mode: Mode,
    /// Subject label (without `sub-`).
    #[arg(long, default_value = "01")]
    sub: String,
    /// Session label (without `ses-`).
    #[arg(long)]
    ses: Option<String>,
    /// Dataset root to write.
    #[arg(short, long, value_name = "DIR")]
    out: PathBuf,
    /// Random seed; overrides the overlay's.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("aslscan: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let t0 = Instant::now();

    let ph = phantom::load(&cli.phantom)?;
    let mut p = protocol::load(&cli.asl_json, &cli.aslcontext, cli.overlay.as_deref(), ph.params.as_ref())?;
    if let Some(s) = cli.seed {
        p.seed = s;
    }
    let mode = match cli.t2_mode {
        Mode::Auto => T2Mode::Auto,
        Mode::Class => T2Mode::Class,
        Mode::Voxel => T2Mode::Voxel,
    };
    println!(
        "protocol: {:?}, {} volumes, M0 {:?}, alpha {} ({}), lambda {} ({}), T1b {} ({}), T2blood {} s ({}), seed {}",
        p.label_type, p.rows.len(), p.m0_type, p.alpha.0, p.alpha.1.as_str(), p.lambda.0, p.lambda.1.as_str(),
        p.t1b.0, p.t1b.1.as_str(), p.t2_blood_s.0, p.t2_blood_s.1.as_str(), p.seed,
    );
    println!("phantom: {:?} voxels, labels {:?}, fieldmap {}", ph.grid.dims,
             ph.labels.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>(), ph.fieldmap.is_some());
    println!(
        "contrast {}{}; background suppression {}; motion {}",
        p.contrast.as_str(),
        p.ir.as_ref().map_or(String::new(), |s| format!(
            " (TI {} s, flip {} deg, inversion {} deg)",
            s.params.inversion_time, s.params.excitation_flip_deg, s.params.inversion_flip_deg)),
        p.suppression.as_ref().map_or("off".to_string(), |s| format!(
            "{} pulse(s) on row 0 at efficiency {}{}", s.per_row[0].len(), s.epsilon.0,
            if s.presaturation.0 { " with presaturation" } else { "" })),
        p.motion.as_ref().map_or("off".to_string(), |m| format!(
            "{}{}", m.mode_name, if m.within.is_some() { " + within-volume events" } else { "" })),
    );

    let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
    let t1 = Instant::now();
    let out = series::simulate(&p, &ph, mode, &phase)?;
    println!(
        "simulated: acquisition {:?}, simulation {:?}, T2 mode {}, {} compartments, {:.1?}",
        out.acq_grid.dims, out.sim_grid.dims, out.mode.as_str(), out.n_compartments, t1.elapsed(),
    );

    let names = Names::new(&cli.sub, cli.ses.as_deref());
    write_dataset(&cli.out, &names, &p, &out)?;
    println!("wrote {} ({:.1?} total)", cli.out.join(names.rel("_part-{mag,phase}_asl.nii.gz")).display(), t0.elapsed());
    Ok(())
}
