"""
LeapfrogDose - CT Dose Analytics & Leapfrog Reporting Tool
===========================================================
Built for diagnostic medical physicists providing dose data 
analysis and Leapfrog survey preparation services.

Author: Daniel Soliman
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# LEAPFROG BENCHMARK DATA
# ============================================================
# ACR DIR / Leapfrog reference values for CT DLP (mGy·cm)
# Sources:
#   - ACR Dose Index Registry (DIR) 2023-2024 national reference levels
#   - AAPM Report 204 (Size-Specific Dose Estimates)
#   - Leapfrog Group Hospital Survey Section 8B specifications
#
# Note: These are reference benchmarks for quality improvement purposes.
# For official ACR DIR registry submission, obtain current subscription data.
# Values represent typical national medians and are updated annually.

ADULT_BENCHMARKS = {
    "Head": {"p25": 800, "p50": 950, "p75": 1100, "achievable": 750},
    "Chest": {"p25": 200, "p50": 350, "p75": 500, "achievable": 200},
    "Abdomen-Pelvis": {"p25": 400, "p50": 600, "p75": 900, "achievable": 400},
    "Chest-Abdomen-Pelvis": {"p25": 500, "p50": 800, "p75": 1200, "achievable": 500},
    "Spine": {"p25": 400, "p50": 600, "p75": 900, "achievable": 400},
}

# Pediatric benchmarks by age group (DLP in mGy·cm)
# Based on ACR DIR pediatric reference levels
PEDS_BENCHMARKS = {
    "Head": {
        "<1":    {"p25": 200, "p50": 300, "p75": 400},
        "1-4":   {"p25": 300, "p50": 400, "p75": 550},
        "5-9":   {"p25": 400, "p50": 550, "p75": 700},
        "10-14": {"p25": 500, "p50": 700, "p75": 900},
        "15-17": {"p25": 600, "p50": 800, "p75": 1000},
    },
    "Abdomen-Pelvis": {
        "<1":    {"p25": 50,  "p50": 100, "p75": 200},
        "1-4":   {"p25": 80,  "p50": 150, "p75": 250},
        "5-9":   {"p25": 100, "p50": 200, "p75": 350},
        "10-14": {"p25": 150, "p50": 300, "p75": 500},
        "15-17": {"p25": 250, "p50": 450, "p75": 700},
    },
}

# Leapfrog age strata (exact labels per Leapfrog Survey specification)
PEDS_AGE_GROUPS = ["<1", "1-4", "5-9", "10-14", "15-17"]


# ============================================================
# BODY REGION CLASSIFICATION
# ============================================================

# Maps common Radimetrics study descriptions / protocol names
# to standardized body regions. Add more mappings as needed.
REGION_PATTERNS = {
    "Head": [
        "head", "brain", "cranial", "skull", "sinus", "orbit",
        "facial", "temporal", "posterior fossa",
    ],
    "Chest": [
        "chest", "thorax", "lung", "pulmonary", "pe study",
        "ct angio chest", "hrct", "ld chest",
    ],
    "Abdomen-Pelvis": [
        "abd", "abdomen", "pelvis", "abdominal", "a/p", "a&p",
        "abdpel", "renal", "liver", "pancrea", "kidney",
    ],
    "Chest-Abdomen-Pelvis": [
        "cap", "chest abd pelvis", "chest abdomen pelvis",
        "chest/abd/pelvis", "chest abdomen", "chest abd",
        "torso", "whole body",
    ],
    "Spine": [
        "spine", "cervical", "thoracic", "lumbar", "c-spine",
        "t-spine", "l-spine",
    ],
}


def classify_body_region(study_desc: str, scan_region: str = "") -> str:
    """
    Classify a CT exam into a standardized body region based on 
    study description and scan region fields from Radimetrics.
    
    Returns the matched region or 'Other' if no match found.
    """
    text = f"{study_desc} {scan_region}".lower().strip()
    
    # Check CAP first (most specific, before chest or abd match)
    for pattern in REGION_PATTERNS["Chest-Abdomen-Pelvis"]:
        if pattern in text:
            return "Chest-Abdomen-Pelvis"
    
    # Then check remaining regions
    for region in ["Head", "Chest", "Abdomen-Pelvis", "Spine"]:
        for pattern in REGION_PATTERNS[region]:
            if pattern in text:
                return region
    
    return "Other"


def classify_age_group(age_years: float) -> str:
    """Classify patient age into Leapfrog pediatric age strata.
    
    Leapfrog specifies: <1, 1-4, 5-9, 10-14, 15-17
    Boundaries: <1 means 0-0.999, 1-4 means 1.0-4.999, etc.
    """
    if age_years < 1:
        return "<1"
    elif age_years < 5:
        return "1-4"
    elif age_years < 10:
        return "5-9"
    elif age_years < 15:
        return "10-14"
    elif age_years < 18:
        return "15-17"
    else:
        return "Adult"


def is_routine_exam(study_desc: str, region: str) -> bool:
    """
    Determine if an exam is 'routine' (non-contrast, single-phase)
    vs. specialized (CTA, perfusion, multi-phase, etc.).

    Leapfrog Section 8B specifically asks for routine exams only.
    """
    desc_lower = study_desc.lower().strip()

    # Non-routine indicators
    nonroutine_terms = [
        # Angiography
        "cta", "ct angio", "angiogram", "angiography", "runoff",

        # Contrast/multiphase
        "3 phase", "triphasic", "multiphase", "multi-phase",
        "pre and post", "pre+post", "pre & post", "arterial", "venous",
        "portal venous", "delayed",

        # Specialized protocols
        "perfusion", "ct perf", "stone protocol", "renal stone",
        "low dose", "ultra low", "lung screening", "calcium score",
        "cardiac", "coronary",

        # Interventional
        "biopsy", "drainage", "guided",
    ]

    for term in nonroutine_terms:
        if term in desc_lower:
            return False

    return True


def is_multiphase_exam(study_desc: str) -> bool:
    """
    Detect if an exam is likely multi-phase based on study description.
    Multi-phase exams have proportionally higher DLP and should be
    flagged for review.
    """
    desc_lower = study_desc.lower().strip()

    multiphase_terms = [
        "3 phase", "triphasic", "tri-phasic", "multiphase", "multi-phase",
        "pre and post", "pre+post", "pre & post",
        "arterial", "venous", "portal venous", "delayed",
        "pre contrast", "post contrast", "pre/post",
    ]

    return any(term in desc_lower for term in multiphase_terms)


# ============================================================
# DATA LOADING & PARSING
# ============================================================

# Expected column mappings from Radimetrics CSV exports
# Maps possible Radimetrics column names -> our standardized names
COLUMN_MAPPINGS = {
    # Patient identifiers
    "patient_id": ["patient id", "patientid", "patient_id", "mrn",
                   "accession", "accession number", "study id"],

    # Demographics
    "age": ["age", "patient age", "age (years)", "age_years",
            "patient_age", "age at exam"],
    "sex": ["sex", "gender", "patient sex", "patient_sex"],

    # Exam info
    "exam_date": ["exam date", "study date", "date", "exam_date",
                  "study_date", "acquisition date"],
    "study_description": ["study description", "study_description",
                          "exam description", "protocol", "protocol name",
                          "study_desc", "exam_desc", "description"],
    "scan_region": ["scan region", "body region", "body_region",
                    "scan_region", "region", "anatomic region",
                    "anatomy", "body part"],

    # Scanner info
    "scanner": ["scanner", "station name", "device", "equipment",
                "scanner_model", "ct scanner", "station"],
    "manufacturer": ["manufacturer", "scanner manufacturer"],

    # Dose metrics
    "ctdivol": ["ctdivol", "ctdi_vol", "ctdi vol", "ctdi",
                "volume ctdi", "ctdivol (mgy)", "ctdivol_mgy"],
    "dlp": ["dlp", "dose length product", "dlp (mgy cm)",
            "dlp (mgy·cm)", "dlp_mgy_cm", "total dlp"],
    "ssde": ["ssde", "size specific dose", "ssde (mgy)",
             "ssde_mgy"],
    "effective_dose": ["effective dose", "eff dose", "ed",
                       "effective_dose", "ed (msv)", "eff_dose_msv"],

    # Phantom size
    "phantom": ["phantom", "phantom size", "ctdi phantom",
                "phantom_size", "phantom diameter"],
}


def find_column(df_columns: list, field_name: str) -> str:
    """
    Find the matching column in the dataframe for a given field.
    Returns the matched column name or None.
    """
    possible = COLUMN_MAPPINGS.get(field_name, [])
    df_cols_lower = {c.lower().strip(): c for c in df_columns}

    for p in possible:
        if p in df_cols_lower:
            return df_cols_lower[p]

    return None


def load_radimetrics_csv(filepath: str) -> pd.DataFrame:
    """
    Load and standardize a Radimetrics CSV export.

    Handles common variations in column naming and formatting.
    Returns a DataFrame with standardized column names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try reading with common encodings and delimiters
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        for sep in [',', '\t', ';']:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                if len(df.columns) > 3:  # Reasonable number of columns
                    break
            except Exception:
                continue
        else:
            continue
        break
    else:
        raise ValueError(f"Could not parse {filepath}. Check format.")

    # Map columns to standardized names
    col_map = {}
    for field in COLUMN_MAPPINGS:
        matched = find_column(df.columns.tolist(), field)
        if matched:
            col_map[matched] = field

    df = df.rename(columns=col_map)

    # Parse age to numeric
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Parse DLP to numeric
    if 'dlp' in df.columns:
        df['dlp'] = pd.to_numeric(df['dlp'], errors='coerce')

    # Parse CTDIvol to numeric
    if 'ctdivol' in df.columns:
        df['ctdivol'] = pd.to_numeric(df['ctdivol'], errors='coerce')

    # Parse exam date
    if 'exam_date' in df.columns:
        df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')

    # Classify body region
    study_col = 'study_description' if 'study_description' in df.columns else None
    region_col = 'scan_region' if 'scan_region' in df.columns else None

    if study_col or region_col:
        df['body_region'] = df.apply(
            lambda row: classify_body_region(
                str(row.get(study_col, '')),
                str(row.get(region_col, ''))
            ), axis=1
        )

    # Classify age group
    if 'age' in df.columns:
        df['age_group'] = df['age'].apply(classify_age_group)
        df['is_pediatric'] = df['age'] < 18

    # Classify routine vs. non-routine
    if study_col:
        df['is_routine'] = df.apply(
            lambda row: is_routine_exam(
                str(row.get(study_col, '')),
                row.get('body_region', '')
            ), axis=1
        )
        df['is_multiphase'] = df.apply(
            lambda row: is_multiphase_exam(
                str(row.get(study_col, ''))
            ), axis=1
        )

    # Phantom size validation (if available)
    if 'phantom' in df.columns:
        df['phantom_size'] = df['phantom'].str.extract(r'(\d+)').astype(float)

    # Drop rows with no DLP data
    if 'dlp' in df.columns:
        n_before = len(df)
        df = df.dropna(subset=['dlp'])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} rows with missing DLP values")

    # Drop rows with DLP <= 0 (invalid or scout/localizer)
    if 'dlp' in df.columns:
        df = df[df['dlp'] > 0]

    print(f"  Loaded {len(df)} valid CT dose records")
    print(f"  Date range: {df['exam_date'].min()} to {df['exam_date'].max()}"
          if 'exam_date' in df.columns else "")
    print(f"  Body regions found: {df['body_region'].value_counts().to_dict()}"
          if 'body_region' in df.columns else "")

    return df


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def compute_percentiles(series: pd.Series) -> dict:
    """Compute 25th, 50th, 75th percentiles and stats for a dose series."""
    if len(series) < 1:
        return None
    return {
        "n": len(series),
        "mean": round(series.mean(), 1),
        "std": round(series.std(), 1),
        "min": round(series.min(), 1),
        "p25": round(series.quantile(0.25), 1),
        "p50": round(series.quantile(0.50), 1),
        "p75": round(series.quantile(0.75), 1),
        "max": round(series.max(), 1),
    }


def flag_outliers(df: pd.DataFrame, region: str,
                  multiplier: float = 2.0) -> pd.DataFrame:
    """
    Flag individual exams as outliers if DLP exceeds
    multiplier × 75th percentile for that body region.
    """
    subset = df[df['body_region'] == region].copy()
    if len(subset) < 4:
        return subset

    p75 = subset['dlp'].quantile(0.75)
    threshold = p75 * multiplier
    subset['is_outlier'] = subset['dlp'] > threshold
    subset['outlier_threshold'] = threshold

    return subset


def compare_to_benchmarks(facility_stats: dict, region: str,
                          age_group: str = None) -> dict:
    """
    Compare facility percentiles against national benchmarks.
    Returns benchmark data and pass/flag status.
    """
    if age_group and age_group != "Adult":
        benchmarks = PEDS_BENCHMARKS.get(region, {}).get(age_group)
    else:
        benchmarks = ADULT_BENCHMARKS.get(region)

    if not benchmarks or not facility_stats:
        return {"benchmarks": None, "status": "No benchmark available"}

    result = {
        "benchmarks": benchmarks,
        "facility": facility_stats,
    }

    # Status logic
    fac_p75 = facility_stats["p75"]
    bench_p75 = benchmarks["p75"]

    if fac_p75 <= benchmarks.get("achievable", benchmarks["p25"]):
        result["status"] = "EXCELLENT"
        result["detail"] = "At or below achievable dose level"
    elif fac_p75 <= benchmarks["p50"]:
        result["status"] = "GOOD"
        result["detail"] = "Below national median"
    elif fac_p75 <= bench_p75:
        result["status"] = "ACCEPTABLE"
        result["detail"] = "Between median and 75th percentile"
    else:
        result["status"] = "ABOVE BENCHMARK"
        result["detail"] = (f"Facility 75th %ile ({fac_p75:.0f}) exceeds "
                           f"national 75th %ile ({bench_p75:.0f})")

    # Percent difference from national median
    bench_p50 = benchmarks["p50"]
    fac_p50 = facility_stats["p50"]
    pct_diff = ((fac_p50 - bench_p50) / bench_p50) * 100
    result["pct_vs_national_median"] = round(pct_diff, 1)

    return result


def analyze_facility(df: pd.DataFrame, facility_name: str = "Facility") -> dict:
    """
    Run full Leapfrog analysis on a facility's dose data.

    Returns a structured results dictionary with:
    - Adult results by body region
    - Pediatric results by body region and age group
    - Outlier flags
    - Benchmark comparisons
    - Summary statistics
    """
    results = {
        "facility_name": facility_name,
        "analysis_date": datetime.now().isoformat(),
        "total_exams": len(df),
        "date_range": {
            "start": str(df['exam_date'].min()) if 'exam_date' in df.columns else None,
            "end": str(df['exam_date'].max()) if 'exam_date' in df.columns else None,
        },
        "adult": {},
        "pediatric": {},
        "outliers": [],
        "scanners": [],
    }

    # Scanner inventory
    if 'scanner' in df.columns:
        results["scanners"] = df['scanner'].unique().tolist()

    # ---------- ADULT ANALYSIS ----------
    adult_df = df[df['age_group'] == 'Adult'] if 'age_group' in df.columns else df

    for region in ["Head", "Chest", "Abdomen-Pelvis",
                   "Chest-Abdomen-Pelvis", "Spine"]:
        region_df = adult_df[adult_df['body_region'] == region]

        if len(region_df) < 1:
            continue

        # Separate routine vs. all exams
        routine_df = region_df[region_df.get('is_routine', True) == True] if 'is_routine' in region_df.columns else region_df

        # Compute stats for routine exams (Leapfrog requirement)
        stats = compute_percentiles(routine_df['dlp'])
        comparison = compare_to_benchmarks(stats, region)

        # Count breakdowns
        n_total = len(region_df)
        n_routine = len(routine_df)
        n_nonroutine = n_total - n_routine
        n_multiphase = len(region_df[region_df.get('is_multiphase', False) == True]) if 'is_multiphase' in region_df.columns else 0

        results["adult"][region] = {
            "stats": stats,
            "benchmark_comparison": comparison,
            "exam_counts": {
                "total": n_total,
                "routine": n_routine,
                "non_routine": n_nonroutine,
                "multiphase": n_multiphase,
            },
        }

        # Flag outliers for THIS region (inside loop)
        outlier_df = flag_outliers(adult_df, region)
        outliers = outlier_df[outlier_df.get('is_outlier', False) == True]
        if len(outliers) > 0:
            for _, row in outliers.iterrows():
                results["outliers"].append({
                    "population": "Adult",
                    "region": region,
                    "dlp": row['dlp'],
                    "threshold": row.get('outlier_threshold', None),
                    "exam_date": str(row.get('exam_date', '')),
                    "study_description": str(row.get('study_description', '')),
                    "patient_id": str(row.get('patient_id', '')),
                })

    # ---------- PEDIATRIC ANALYSIS ----------
    if 'is_pediatric' in df.columns:
        peds_df = df[df['is_pediatric'] == True]

        for region in ["Head", "Abdomen-Pelvis"]:
            region_df = peds_df[peds_df['body_region'] == region]

            if len(region_df) < 1:
                continue

            results["pediatric"][region] = {}

            for age_grp in PEDS_AGE_GROUPS:
                age_df = region_df[region_df['age_group'] == age_grp]

                if len(age_df) < 1:
                    continue

                # Filter to routine exams only
                routine_df = age_df[age_df.get('is_routine', True) == True] if 'is_routine' in age_df.columns else age_df

                stats = compute_percentiles(routine_df['dlp'])
                comparison = compare_to_benchmarks(stats, region, age_grp)

                # Leapfrog requires minimum 10 exams per category for scoring
                low_n = stats["n"] < 10

                n_total = len(age_df)
                n_routine = len(routine_df)

                results["pediatric"][region][age_grp] = {
                    "stats": stats,
                    "benchmark_comparison": comparison,
                    "low_n": low_n,
                    "leapfrog_eligible": not low_n,
                    "exam_counts": {
                        "total": n_total,
                        "routine": n_routine,
                        "non_routine": n_total - n_routine,
                    },
                }

    # ---------- SUMMARY ----------
    n_above = sum(
        1 for r in results["adult"].values()
        if r["benchmark_comparison"]["status"] == "ABOVE BENCHMARK"
    )
    results["summary"] = {
        "total_adult_exams": len(adult_df) if 'age_group' in df.columns else len(df),
        "total_peds_exams": len(df[df['is_pediatric'] == True]) if 'is_pediatric' in df.columns else 0,
        "regions_analyzed": len(results["adult"]),
        "regions_above_benchmark": n_above,
        "total_outliers": len(results["outliers"]),
    }

    return results


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_report(results: dict):
    """Print a formatted console report of the analysis results."""

    print("\n" + "=" * 70)
    print(f"  LEAPFROG CT DOSE ANALYSIS REPORT")
    print(f"  {results['facility_name']}")
    print("=" * 70)
    print(f"  Analysis Date:  {results['analysis_date'][:10]}")
    print(f"  Data Range:     {results['date_range']['start'][:10] if results['date_range']['start'] else 'N/A'}"
          f" to {results['date_range']['end'][:10] if results['date_range']['end'] else 'N/A'}")
    print(f"  Total Exams:    {results['total_exams']}")
    print(f"  Scanners:       {', '.join(results['scanners']) if results['scanners'] else 'N/A'}")
    print("-" * 70)

    # Adult results
    print(f"\n  ADULT CT DOSE RESULTS (DLP, mGy·cm)")
    print(f"  {'Region':<25} {'N':>5} {'Rtn':>5} {'P25':>8} {'P50':>8} {'P75':>8} {'Status':<18}")
    print(f"  {'-'*23:<25} {'---':>5} {'---':>5} {'---':>8} {'---':>8} {'---':>8} {'-'*16:<18}")

    for region, data in results["adult"].items():
        s = data["stats"]
        counts = data.get("exam_counts", {})
        n_total = counts.get("total", s['n'])
        n_routine = counts.get("routine", s['n'])

        status = data["benchmark_comparison"]["status"]
        status_icon = {
            "EXCELLENT": "✓ EXCELLENT",
            "GOOD": "✓ GOOD",
            "ACCEPTABLE": "~ ACCEPTABLE",
            "ABOVE BENCHMARK": "✗ ABOVE",
        }.get(status, status)

        print(f"  {region:<25} {n_total:>5} {n_routine:>5} {s['p25']:>8.0f} {s['p50']:>8.0f} "
              f"{s['p75']:>8.0f} {status_icon:<18}")

    # Legend
    print(f"  {'':>25} {'(all)':>5} {'(rtn)':>5}")
    print(f"  N = total exams, Rtn = routine exams only (used for benchmarking)")

    # Pediatric results
    if results["pediatric"]:
        print(f"\n  PEDIATRIC CT DOSE RESULTS (DLP, mGy·cm)")
        print(f"  {'Region':<18} {'Age':>6} {'N':>5} {'P25':>8} {'P50':>8} {'P75':>8} {'Status':<18}")
        print(f"  {'-'*16:<18} {'---':>6} {'---':>5} {'---':>8} {'---':>8} {'---':>8} {'-'*16:<18}")

        for region, age_data in results["pediatric"].items():
            for age_grp, data in age_data.items():
                s = data["stats"]
                status = data["benchmark_comparison"]["status"]
                status_icon = {
                    "EXCELLENT": "✓ EXCELLENT",
                    "GOOD": "✓ GOOD",
                    "ACCEPTABLE": "~ ACCEPTABLE",
                    "ABOVE BENCHMARK": "✗ ABOVE",
                }.get(status, status)

                print(f"  {region:<18} {age_grp:>6} {s['n']:>5} {s['p25']:>8.0f} "
                      f"{s['p50']:>8.0f} {s['p75']:>8.0f} {status_icon:<18}")

    # Outliers
    if results["outliers"]:
        print(f"\n  FLAGGED OUTLIERS ({len(results['outliers'])} exams)")
        print(f"  {'Region':<22} {'DLP':>8} {'Threshold':>10} {'Date':<12} {'Description'}")
        print(f"  {'-'*20:<22} {'---':>8} {'---':>10} {'---':<12} {'-'*20}")

        for o in results["outliers"][:20]:  # Show top 20
            print(f"  {o['region']:<22} {o['dlp']:>8.0f} {o['threshold']:>10.0f} "
                  f"{o['exam_date'][:10]:<12} {o['study_description'][:30]}")

        if len(results["outliers"]) > 20:
            print(f"  ... and {len(results['outliers']) - 20} more")

    # Data quality notes
    print(f"\n  DATA QUALITY NOTES")
    print(f"  {'-'*40}")

    total_nonroutine = sum(
        data.get("exam_counts", {}).get("non_routine", 0)
        for data in results["adult"].values()
    )
    total_multiphase = sum(
        data.get("exam_counts", {}).get("multiphase", 0)
        for data in results["adult"].values()
    )

    if total_nonroutine > 0:
        print(f"  Non-routine exams excluded:  {total_nonroutine}")
        print(f"    (CTA, perfusion, specialized protocols)")

    if total_multiphase > 0:
        print(f"  Multi-phase exams flagged:   {total_multiphase}")
        print(f"    (May contribute to outliers)")

    print(f"  Leapfrog uses routine-only exams for percentile calculations.")

    # Summary
    print(f"\n  SUMMARY")
    print(f"  {'-'*40}")
    s = results["summary"]
    print(f"  Adult exams analyzed:       {s['total_adult_exams']}")
    print(f"  Pediatric exams analyzed:   {s['total_peds_exams']}")
    print(f"  Body regions analyzed:      {s['regions_analyzed']}")
    print(f"  Regions above benchmark:    {s['regions_above_benchmark']}")
    print(f"  Outlier exams flagged:      {s['total_outliers']}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def run_analysis(csv_path: str, facility_name: str = "Facility") -> dict:
    """
    Main entry point. Load CSV, analyze, print report, return results.

    Usage:
        results = run_analysis("dose_data.csv", "Memorial Hospital")
    """
    print(f"\nLoading data from {csv_path}...")
    df = load_radimetrics_csv(csv_path)

    print(f"\nAnalyzing dose data for {facility_name}...")
    results = analyze_facility(df, facility_name)

    print_report(results)

    return results, df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python leapfrog_dose.py <csv_file> [facility_name]")
        print("       python leapfrog_dose.py --demo")
        sys.exit(1)

    if sys.argv[1] == "--demo":
        print("Run generate_sample_data.py first, then:")
        print("  python leapfrog_dose.py sample_dose_data.csv 'Demo Hospital'")
    else:
        name = sys.argv[2] if len(sys.argv) > 2 else "Facility"
        run_analysis(sys.argv[1], name)