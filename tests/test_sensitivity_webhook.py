"""
Five known parameter combinations → assert correct classification via the webhook endpoint.
These are proof-of-engine tests: if the physics math is wrong, these fail.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from web.app import app
from web.models import init_db

init_db()  # create tables before any request hits the DB
client = TestClient(app)

CASES = [
    # (description, payload, expected_classification)
    (
        "optimal acquisition — no meaningful degradation",
        {
            "StudyInstanceUID": "1.2.3.GREEN",
            "SliceThickness": 1.25,
            "ConvolutionKernel": "B30f",
            "CTDIvol": 9.0,
            "AIModelVersion": "1.0.0",
        },
        "GREEN",
    ),
    (
        "soft kernel alone — moderate degradation",
        {
            "StudyInstanceUID": "1.2.3.YELLOW.KERNEL",
            "SliceThickness": 1.25,
            "ConvolutionKernel": "B40f",
            "CTDIvol": 10.0,
            "AIModelVersion": "1.0.0",
        },
        "YELLOW",   # ΔK = -7.9pp
    ),
    (
        "thick slice + LDCT dose — compounding degradation",
        {
            "StudyInstanceUID": "1.2.3.YELLOW.COMPOUND",
            "SliceThickness": 3.75,
            "ConvolutionKernel": "B30f",
            "CTDIvol": 7.5,
            "AIModelVersion": "1.0.0",
        },
        "YELLOW",   # ΔS = -8.1pp, ΔD = -1.2pp → 9.3pp
    ),
    (
        "worst case — 5mm slice + soft kernel",
        {
            "StudyInstanceUID": "1.2.3.RED.WORST",
            "SliceThickness": 5.0,
            "ConvolutionKernel": "B40f",
            "CTDIvol": 8.2,
            "ManufacturerModelName": "Siemens SOMATOM",
            "AIModelVersion": "1.0.0",
        },
        "RED",      # ΔS = -13.2pp, ΔK = -7.9pp → 22pp
    ),
    (
        "thick slice alone crosses RED threshold",
        {
            "StudyInstanceUID": "1.2.3.RED.SLICE",
            "SliceThickness": 5.0,
            "ConvolutionKernel": "B30f",
            "CTDIvol": 10.0,
            "AIModelVersion": "1.0.0",
        },
        "RED",      # ΔS = -13.2pp alone
    ),
]


@pytest.mark.parametrize("description,payload,expected", CASES)
def test_webhook_classification(description, payload, expected):
    r = client.post("/api/sensitivity/dicom/study", json=payload)
    assert r.status_code == 200, f"{description}: HTTP {r.status_code} — {r.text}"
    body = r.json()
    assert body["classification"] == expected, (
        f"{description}: got {body['classification']} "
        f"(degradation {body['degradation_pp']}pp), expected {expected}"
    )
