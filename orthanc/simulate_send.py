"""
Simulates what the Orthanc plugin sends to GammaMetric.
Use this to test the full pipeline without running Orthanc.

Usage:
  python simulate_send.py --preset red
  python simulate_send.py --preset yellow --url http://localhost:8000/api/sensitivity/dicom/study
  python simulate_send.py --slice 5.0 --kernel B30f --ctdi 5.0 --scanner "GE Revolution"
"""

import argparse
import json
import uuid
import requests

DEFAULT_URL = "https://dose.gammametric.com/api/sensitivity/dicom/study"

PRESETS = {
    "green":  {"SliceThickness": 1.25, "ConvolutionKernel": "B30f",  "CTDIvol": 10.0},
    "yellow": {"SliceThickness": 3.0,  "ConvolutionKernel": "B30f",  "CTDIvol": 7.5},
    "red":    {"SliceThickness": 5.0,  "ConvolutionKernel": "B30f",  "CTDIvol": 5.0},
    "soft":   {"SliceThickness": 5.0,  "ConvolutionKernel": "SOFT",  "CTDIvol": 7.5},
}


def main():
    parser = argparse.ArgumentParser(description="Simulate Orthanc → GammaMetric webhook")
    parser.add_argument("--preset",  choices=PRESETS.keys(), default="red",
                        help="Preset acquisition scenario (default: red)")
    parser.add_argument("--slice",   type=float, help="Override SliceThickness (mm)")
    parser.add_argument("--kernel",  help="Override ConvolutionKernel")
    parser.add_argument("--ctdi",    type=float, help="Override CTDIvol (mGy)")
    parser.add_argument("--scanner", default="Siemens SOMATOM Force",
                        help="ManufacturerModelName")
    parser.add_argument("--url",     default=DEFAULT_URL, help="Endpoint URL")
    parser.add_argument("--version", default="v1.0.0", help="AI model version")
    args = parser.parse_args()

    base = dict(PRESETS[args.preset])
    if args.slice:  base["SliceThickness"]     = args.slice
    if args.kernel: base["ConvolutionKernel"]  = args.kernel
    if args.ctdi:   base["CTDIvol"]            = args.ctdi

    payload = {
        "StudyInstanceUID":    f"1.2.826.0.1.3680043.sim.{uuid.uuid4().hex[:8]}",
        "SliceThickness":      base["SliceThickness"],
        "ConvolutionKernel":   base["ConvolutionKernel"],
        "CTDIvol":             base["CTDIvol"],
        "ManufacturerModelName": args.scanner,
        "KVP":                 120,
        "AcquisitionDate":     "20260515",
        "AIModelVersion":      args.version,
    }

    print(f"POST {args.url}")
    print(json.dumps(payload, indent=2))
    print()

    resp = requests.post(args.url, json=payload, timeout=15)
    resp.raise_for_status()
    result = resp.json()

    cls   = result["classification"]
    delta = result["degradation_pp"]
    alert = result["alert_sent"]
    ood   = result["out_of_distribution"]

    print(f"Classification : {cls}")
    print(f"Degradation    : {delta:+.1f} pp")
    print(f"OOD            : {ood}")
    print(f"Alert sent     : {alert}")


if __name__ == "__main__":
    main()
