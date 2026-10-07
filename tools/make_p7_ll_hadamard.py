#!/usr/bin/env python3
"""Write tests/fixtures/protocols/p7_ll_hadamard (P7 plan, Task 8): Hadamard-8 PCASL on the crop, each
encoded preparation read by four Look-Locker readouts. Seven sub-boli of 0.25 s, PLD_n = 0.5, 0.8,
1.1, 1.4 s, 35 degrees, TR 4.5 s, two encoding cycles after an included M0; exchange on. The context
lists the decoded volumes readout-major: row (j, n) has delay PLD_n + the later sub-boli.

    make_p7_ll_hadamard.py <fixture dir>
"""
import json
import sys
from pathlib import Path

H, TAU, PLDS, CYCLES = 8, 0.25, [0.5, 0.8, 1.1, 1.4], 2
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
ld, pld, kinds = [0.0], [0.0], ["m0scan"]
for _ in range(CYCLES):
    for p_n in PLDS:
        for j in range(H - 1):
            ld.append(TAU)
            pld.append(round(p_n + TAU * (H - 2 - j), 6))
            kinds.append("deltam")
side = {
    "Manufacturer": "aslscan-fixture", "MagneticFieldStrength": 3, "MRAcquisitionType": "2D",
    "PulseSequenceType": "EPI", "PhaseEncodingDirection": "j-", "ArterialSpinLabelingType": "PCASL",
    "BackgroundSuppression": False, "VascularCrushing": False, "LookLocker": True, "FlipAngle": 35,
    "LabelingDuration": ld, "PostLabelingDelay": pld, "M0Type": "Included", "RepetitionTimePreparation": 4.5,
    "EchoTime": 0.012, "TotalReadoutTime": 0.012, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "SliceTiming": [0.0, 0.05],
}
(out / "asl.json").write_text(json.dumps(side, indent=1) + "\n")
(out / "aslcontext.tsv").write_text("volume_type\n" + "".join(f"{k}\n" for k in kinds))
(out / "overlay.toml").write_text(f"""# The P7 Look-Locker Hadamard acceptance case (P7 plan, Task 8) on the crop: H8 PCASL, seven
# sub-boli of 0.25 s, each encoded preparation read at PLD_n = 0.5, 0.8, 1.1, 1.4 s (35 degrees), TR
# 4.5 s, two encoding cycles after an included M0; exchange on.
seed = 87

[acquisition]
oversample = 2
signal_scale = 100.0
noise_variance = 0.01

[signal]
acq_contrast = "ge"

[kinetic]
exchange_time = 0.5

[hadamard]
order = {H}

[look_locker]
readouts_per_cycle = {len(PLDS)}
""")
