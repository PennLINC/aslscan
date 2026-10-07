#!/usr/bin/env python3
"""Write tests/fixtures/protocols/p7_ll3d (P7 plan, Task 16): 3D Look-Locker PASL on the crop. Each
cycle is six readouts, 0.3 s apart from 0.8 s, each a sub-train of the 3D EPI train (8 degrees, 40 ms
between excitations); two shots (two ky segments), so a group of two cycles gives each readout's
volume. A control group and a label group, TR 4.0 s, the label entering the slab 0.3 s after
labeling, a separate M0.

    make_p7_ll3d.py <fixture dir>
"""
import json
import sys
from pathlib import Path

M = 6
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
pld = [round(0.8 + 0.3 * n, 3) for _ in range(2) for n in range(M)]
side = {
    "Manufacturer": "aslscan-fixture", "MagneticFieldStrength": 3, "MRAcquisitionType": "3D",
    "PulseSequenceType": "3D EPI", "PhaseEncodingDirection": "j-", "ArterialSpinLabelingType": "PASL",
    "BolusCutOffFlag": True, "BolusCutOffTechnique": "Q2TIPS", "BolusCutOffDelayTime": 0.7,
    "BackgroundSuppression": False, "VascularCrushing": False, "LookLocker": True, "FlipAngle": 8,
    "PostLabelingDelay": pld, "M0Type": "Separate", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012,
    "EffectiveEchoSpacing": 0.0005, "NumberShots": 2, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
}
(out / "asl.json").write_text(json.dumps(side, indent=1) + "\n")
(out / "aslcontext.tsv").write_text("volume_type\n" + "control\n" * M + "label\n" * M)
(out / "overlay.toml").write_text(f"""# The P7 3D Look-Locker acceptance case (P7 plan, Task 16) on the crop: PASL (Q2TIPS, 0.7 s), cycles of
# {M} readouts 0.3 s apart from 0.8 s, each a sub-train of the 3D EPI train (8 degrees, 40 ms between
# excitations), two shots per group, TR 4.0 s, the label entering the slab 0.3 s after labeling.
seed = 89

[acquisition]
oversample = 2
signal_scale = 100.0
noise_variance = 0.01

[signal]
acq_contrast = "ge"

[readout]
excitation_spacing = 40.0
slab_entry_time = 0.3

[look_locker]
readouts_per_cycle = {M}

[m0]
repetition_time = 6.0
""")
