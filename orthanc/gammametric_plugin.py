"""
GammaMetric Orthanc plugin.

Fires on every stable CT study, extracts acquisition tags, and POSTs to the
GammaMetric sensitivity engine. RED studies trigger an immediate alert email;
YELLOW studies are batched into the daily digest.

Install:
  1. Enable the Orthanc Python plugin (libOrthancPython.so / OrthancPython.dll).
  2. Set "Python" -> "Path" in orthanc.json to this file.
  3. Set GAMMAMETRIC_URL and GAMMAMETRIC_API_KEY in orthanc.json (see template).
"""

import json
import urllib.request
import urllib.error

import orthanc  # provided by Orthanc at runtime


# ── Config (injected from orthanc.json "GammaMetric" section) ─────────────────

def _cfg(key, default=None):
    try:
        return json.loads(orthanc.GetConfiguration()).get("GammaMetric", {}).get(key, default)
    except Exception:
        return default


ENDPOINT   = _cfg("Url",        "https://dose.gammametric.com/api/sensitivity/dicom/study")
API_KEY    = _cfg("ApiKey",     "")
AI_VERSION = _cfg("AIModelVersion", "v1.0.0")

# Only process these modalities (extend if needed for MR/PET AI products)
MODALITIES = {"CT"}


# ── Tag extraction ─────────────────────────────────────────────────────────────

def _get_tags(instance_id: str) -> dict:
    raw = orthanc.RestApiGet(f"/instances/{instance_id}/simplified-tags")
    return json.loads(raw)


def _float_tag(tags: dict, key: str):
    val = tags.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Webhook ────────────────────────────────────────────────────────────────────

def _post(payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Api-Key"] = API_KEY

    req = urllib.request.Request(ENDPOINT, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── Main callback ──────────────────────────────────────────────────────────────

def on_stable_study(study_id, change_type, resource_type):
    try:
        study = json.loads(orthanc.RestApiGet(f"/studies/{study_id}"))

        # Find the first CT series
        ct_series_id = None
        for sid in study.get("Series", []):
            series = json.loads(orthanc.RestApiGet(f"/series/{sid}"))
            if series.get("MainDicomTags", {}).get("Modality") in MODALITIES:
                ct_series_id = sid
                break

        if ct_series_id is None:
            return  # not a CT study

        # Use the first instance for tag extraction
        series  = json.loads(orthanc.RestApiGet(f"/series/{ct_series_id}"))
        inst_id = series["Instances"][0]
        tags    = _get_tags(inst_id)

        slice_mm = _float_tag(tags, "SliceThickness")
        ctdivol  = _float_tag(tags, "CTDIvol")
        kernel   = tags.get("ConvolutionKernel") or tags.get("FilterType") or "unknown"

        if slice_mm is None or ctdivol is None:
            orthanc.LogWarning(
                f"GammaMetric: skipping {study_id} — "
                f"SliceThickness={slice_mm}, CTDIvol={ctdivol}"
            )
            return

        payload = {
            "StudyInstanceUID":    tags.get("StudyInstanceUID", study_id),
            "SliceThickness":      slice_mm,
            "ConvolutionKernel":   kernel,
            "CTDIvol":             ctdivol,
            "ManufacturerModelName": tags.get("ManufacturerModelName", "unknown"),
            "KVP":                 _float_tag(tags, "KVP"),
            "AcquisitionDate":     tags.get("AcquisitionDate"),
            "AIModelVersion":      AI_VERSION,
        }

        result = _post(payload)
        orthanc.LogInfo(
            f"GammaMetric: {result['classification']} | "
            f"Δ{result['degradation_pp']:.1f}pp | "
            f"alert={result['alert_sent']}"
        )

    except urllib.error.URLError as exc:
        orthanc.LogError(f"GammaMetric: webhook failed — {exc}")
    except Exception as exc:
        orthanc.LogError(f"GammaMetric: unexpected error — {exc}")


orthanc.RegisterOnStableStudyCallback(on_stable_study)
