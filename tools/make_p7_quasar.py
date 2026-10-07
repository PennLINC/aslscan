#!/usr/bin/env python3
"""Write tests/fixtures/protocols/p7_quasar (P7 plan, Task 5): a QUASAR-like Look-Locker PASL
series on the crop. Four cycles of thirteen readouts (control, label uncrushed; control, label
crushed at 4 cm/s), 0.3 s apart from 0.8 s, 35 degrees, TR 4.8 s; exchange, the arterial
compartment and per-label arterial velocities; a separate M0.

    make_p7_quasar.py <fixture dir>
"""
import json
import sys
from pathlib import Path

M = 13
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
cycles = [("control", 0.0), ("label", 0.0), ("control", 4.0), ("label", 4.0)]
pld = [round(0.8 + 0.3 * n, 3) for _ in cycles for n in range(M)]
venc = [v for _, v in cycles for _ in range(M)]
side = {
    "Manufacturer": "aslscan-fixture", "MagneticFieldStrength": 3, "MRAcquisitionType": "2D",
    "PulseSequenceType": "EPI", "PhaseEncodingDirection": "j-", "ArterialSpinLabelingType": "PASL",
    "BolusCutOffFlag": True, "BolusCutOffTechnique": "Q2TIPS", "BolusCutOffDelayTime": 0.7,
    "BackgroundSuppression": False, "VascularCrushing": True, "VascularCrushingVENC": venc,
    "LookLocker": True, "FlipAngle": 35, "PostLabelingDelay": pld, "M0Type": "Separate",
    "RepetitionTimePreparation": 4.8, "EchoTime": 0.012, "TotalReadoutTime": 0.012,
    "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "SliceTiming": [0.0, 0.05],
}
(out / "asl.json").write_text(json.dumps(side, indent=1) + "\n")
(out / "aslcontext.tsv").write_text("volume_type\n" + "".join(f"{k}\n" for k, _ in cycles for _ in range(M)))
(out / "overlay.toml").write_text(f"""# The P7 QUASAR acceptance case (P7 plan, Task 5) on the crop: Look-Locker PASL (Q2TIPS, 0.7 s),
# 2D EPI, control and label cycles of {M} gradient-echo readouts at 35 degrees, 0.3 s apart from
# 0.8 s, TR 4.8 s, uncrushed then crushed at 4 cm/s; exchange, the arterial compartment.
seed = 83

[acquisition]
oversample = 2
signal_scale = 100.0
noise_variance = 0.01

[signal]
acq_contrast = "ge"

[look_locker]
readouts_per_cycle = {M}

[kinetic]
exchange_time = 0.6

[macrovascular]
arterial_blood_volume = {{ grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }}
arterial_transit_time = {{ grey_matter = 0.5, white_matter = 0.7, csf = 0.0 }}

[vascular_crushing]
arterial_velocity = {{ grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }}

[m0]
repetition_time = 6.0
""")
