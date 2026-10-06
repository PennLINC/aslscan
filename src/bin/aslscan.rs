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
    /// The BIDS ASL sidecar (`*_asl.json`) describing the protocol. Multi-TE: once per echo, in echo
    /// order, each with that echo's EchoTime (written as the BIDS echo-N layout).
    #[arg(long, value_name = "JSON", required = true)]
    asl_json: Vec<PathBuf>,
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
    /// Pin the acquisition to what simasl (ASLDRO v2.2.0) can express: the same as the
    /// overlay's `[compat] asldro = true`.
    #[arg(long)]
    compat_asldro: bool,
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
    let mut p = protocol::load_with(&cli.asl_json.iter().map(PathBuf::as_path).collect::<Vec<_>>(), &cli.aslcontext, cli.overlay.as_deref(), ph.params.as_ref(), cli.compat_asldro)?;
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
    if let Some(c) = &p.compat {
        println!("compat: ASLDRO, grid origin {}, desired SNR {}", p.grid_origin.as_str(),
                 c.desired_snr.map_or("none".to_string(), |s| s.to_string()));
    }
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

    let mut p4 = Vec::new();
    if let Some(te) = p.exchange_time {
        p4.push(format!("exchange {te} s"));
    }
    if let Some(m) = &p.macrovascular {
        p4.push(format!("arterial compartment (T2 {} s, {})", m.t2_arterial.0, m.t2_arterial.1.as_str()));
    }
    if let Some(c) = &p.crushing {
        p4.push(if c.no_arterial_compartment { "crushing (nothing to act on)".to_string() } else { "crushing".to_string() });
    }
    if p.suppression.as_ref().is_some_and(|s| s.model != protocol::SuppressionModel::GlobalBolus) {
        p4.push("bolus-position suppression".to_string());
    }
    if p.physio.is_some() {
        p4.push("physiological noise".to_string());
    }
    if !p4.is_empty() {
        println!("P4: {}", p4.join(", "));
    }

    let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
    let t1 = Instant::now();
    let out = series::simulate(&p, &ph, mode, &phase)?;
    println!(
        "simulated: acquisition {:?}, simulation {:?}, T2 mode {}, {} compartments, {:.1?}",
        out.acq_grid.dims, out.sim_grid.dims, out.mode.as_str(), out.n_compartments, t1.elapsed(),
    );
    if let (Some(r), Some(sp)) = (&out.readout, out.readout.as_ref().and_then(|r| r.spiral.as_ref())) {
        let segs = out.spiral_segmentation.as_deref().unwrap_or(&[]);
        println!(
            "readout: 3D spiral train, {} shot(s), {} echoes of {:.4} ms, {} interleaves of {} samples over {} ms \
             (dwell {} ms, centre region {:.4} ms), refocusing {} degrees; segmentation L <= {}, bound <= {:.1e}",
            r.n_shots, r.etl, r.esp_ms, sp.interleaves, sp.samples_per_interleaf, sp.readout_ms, sp.dwell_ms, sp.tau_c_ms,
            r.train.refocusing_deg, segs.iter().map(|g| g.l).max().unwrap_or(0),
            segs.iter().map(|g| g.bound).fold(0.0, f64::max),
        );
    } else if let Some(r) = &out.readout {
        println!(
            "readout: 3D {} train, {} shot(s), {} echoes of {:.4} ms, {} lines of {:.4} ms per echo ({}), refocusing {} degrees",
            p.readout.as_ref().map_or("?", |s| s.kind.0.as_str()), r.n_shots, r.etl, r.esp_ms, r.epi, r.t_line_ms,
            r.t_line_source, r.train.refocusing_deg,
        );
    }

    if let Some(m) = &p.multi_te {
        println!(
            "multi-TE: {} echoes at {:?} s, {}", p.echo_times_s.len(), p.echo_times_s,
            m.refocusing_time_ms.map_or("gradient echo".to_string(), |r| format!("spin echo, {} ms reserved per refocusing pulse", r.0)),
        );
    }

    let names = Names::new(&cli.sub, cli.ses.as_deref());
    write_dataset(&cli.out, &names, &p, &out)?;
    let tail = if p.echo_times_s.len() > 1 {
        format!("_echo-{{1..{}}}_part-{{mag,phase}}_asl.nii.gz", p.echo_times_s.len())
    } else {
        "_part-{mag,phase}_asl.nii.gz".to_string()
    };
    println!("wrote {} ({:.1?} total)", cli.out.join(names.rel(&tail)).display(), t0.elapsed());
    Ok(())
}
