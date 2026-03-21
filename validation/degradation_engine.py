"""
GammaMetric — AI Validation Test Bench
Image Degradation Engine v0.1

Applies physics-based degradations to DICOM CT series to simulate
varying dose levels, noise conditions, and acquisition parameters.
Used to assess AI tool robustness across imaging environments.

Dependencies: pydicom, numpy, scipy
"""

import os
import copy
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter
from pathlib import Path


# ─── NOISE SIMULATION ────────────────────────────────────────────────────────

def add_gaussian_noise(pixel_array: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add Gaussian noise to simulate increased image noise from lower mAs.
    
    Parameters:
        pixel_array: 2D numpy array of pixel values (HU or raw)
        sigma: noise standard deviation in HU units
                typical CT noise: 10-30 HU for standard dose
                low dose: 40-80 HU
    
    Returns:
        Noisy pixel array (clipped to int16 range)
    """
    noise = np.random.normal(0, sigma, pixel_array.shape)
    noisy = pixel_array.astype(np.float32) + noise
    return np.clip(noisy, -32768, 32767).astype(np.int16)


def add_poisson_noise(pixel_array: np.ndarray, dose_reduction_factor: float) -> np.ndarray:
    """
    Apply Poisson noise scaling to simulate dose reduction.
    More physically accurate than Gaussian for CT dose simulation.
    
    Parameters:
        pixel_array: 2D numpy array of pixel values
        dose_reduction_factor: fraction of original dose (e.g. 0.5 = 50% dose reduction)
    
    Returns:
        Degraded pixel array
    """
    # Shift to non-negative (HU values can be negative)
    shift = abs(pixel_array.min()) + 1
    shifted = pixel_array.astype(np.float32) + shift

    # Scale counts by dose factor, apply Poisson noise, scale back
    scaled = shifted * dose_reduction_factor
    noisy = np.random.poisson(np.maximum(scaled, 0)).astype(np.float32)
    result = (noisy / dose_reduction_factor) - shift

    return np.clip(result, -32768, 32767).astype(np.int16)


# ─── RESOLUTION DEGRADATION ──────────────────────────────────────────────────

def apply_blur(pixel_array: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to simulate soft reconstruction kernel or
    reduced spatial resolution.
    
    Parameters:
        pixel_array: 2D numpy array
        sigma: blur radius in pixels
                0.5 = subtle softening
                1.5 = significant resolution loss
                3.0 = heavy blur (simulate poor reconstruction)
    
    Returns:
        Blurred pixel array
    """
    blurred = gaussian_filter(pixel_array.astype(np.float32), sigma=sigma)
    return np.clip(blurred, -32768, 32767).astype(np.int16)


# ─── COMBINED DEGRADATION PROFILES ───────────────────────────────────────────

DEGRADATION_PROFILES = {
    "standard":       {"noise_sigma": 0,    "dose_factor": 1.0,  "blur_sigma": 0},
    "low_dose_25":    {"noise_sigma": 30,   "dose_factor": 0.25, "blur_sigma": 0.5},
    "low_dose_50":    {"noise_sigma": 20,   "dose_factor": 0.50, "blur_sigma": 0.3},
    "soft_kernel":    {"noise_sigma": 10,   "dose_factor": 1.0,  "blur_sigma": 1.5},
    "poor_technique": {"noise_sigma": 50,   "dose_factor": 0.25, "blur_sigma": 1.0},
    "ultra_low_dose": {"noise_sigma": 80,   "dose_factor": 0.10, "blur_sigma": 1.0},
}


def apply_profile(pixel_array: np.ndarray, profile_name: str) -> np.ndarray:
    """
    Apply a named degradation profile to a pixel array.
    
    Parameters:
        pixel_array: 2D numpy array
        profile_name: key from DEGRADATION_PROFILES
    
    Returns:
        Degraded pixel array
    """
    if profile_name not in DEGRADATION_PROFILES:
        raise ValueError(f"Unknown profile '{profile_name}'. "
                         f"Available: {list(DEGRADATION_PROFILES.keys())}")

    profile = DEGRADATION_PROFILES[profile_name]
    result = pixel_array.copy()

    if profile["dose_factor"] < 1.0:
        result = add_poisson_noise(result, profile["dose_factor"])

    if profile["noise_sigma"] > 0:
        result = add_gaussian_noise(result, profile["noise_sigma"])

    if profile["blur_sigma"] > 0:
        result = apply_blur(result, profile["blur_sigma"])

    return result


# ─── DICOM I/O ────────────────────────────────────────────────────────────────

def load_dicom_series(series_dir: str) -> list[pydicom.Dataset]:
    """
    Load all DICOM files from a directory, sorted by InstanceNumber.
    
    Parameters:
        series_dir: path to directory containing .dcm files
    
    Returns:
        List of pydicom Dataset objects sorted by slice position
    """
    path = Path(series_dir)
    files = sorted(path.glob("*.dcm"))

    if not files:
        # Try without extension
        files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]

    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: could not read {f.name}: {e}")

    # Sort by InstanceNumber if available
    datasets.sort(key=lambda d: int(getattr(d, 'InstanceNumber', 0)))
    print(f"Loaded {len(datasets)} slices from {series_dir}")
    return datasets


def degrade_series(
    datasets: list[pydicom.Dataset],
    profile_name: str,
    output_dir: str
) -> list[str]:
    """
    Apply a degradation profile to an entire DICOM series and save.
    
    Parameters:
        datasets: list of pydicom Dataset objects
        profile_name: degradation profile to apply
        output_dir: directory to write degraded DICOM files
    
    Returns:
        List of output file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_paths = []

    for i, ds in enumerate(datasets):
        # Deep copy to avoid modifying original
        degraded_ds = copy.deepcopy(ds)

        # Get pixel array in HU (apply rescale slope/intercept)
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        hu_array = (pixel_array * slope + intercept).astype(np.int16)

        # Apply degradation
        degraded_hu = apply_profile(hu_array, profile_name)

        # Convert back to raw pixel values
        raw_degraded = ((degraded_hu - intercept) / slope).astype(np.int16)
        degraded_ds.PixelData = raw_degraded.tobytes()

        # Tag the series as degraded for traceability
        degraded_ds.SeriesDescription = (
            f"{getattr(ds, 'SeriesDescription', 'Series')} "
            f"[GammaMetric:{profile_name}]"
        )

        # Write output
        filename = f"degraded_{profile_name}_{i+1:04d}.dcm"
        out_path = os.path.join(output_dir, filename)
        degraded_ds.save_as(out_path)
        output_paths.append(out_path)

    print(f"Saved {len(output_paths)} degraded slices to {output_dir}")
    return output_paths


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

def run_test(input_dir: str, output_base_dir: str):
    """
    Load a series and generate all degradation profiles for comparison.
    
    Parameters:
        input_dir: path to original DICOM series
        output_base_dir: base directory for degraded outputs
    """
    datasets = load_dicom_series(input_dir)

    if not datasets:
        print("No DICOM files found.")
        return

    print(f"\nGenerating degradation profiles for {len(datasets)} slices...")

    for profile_name in DEGRADATION_PROFILES:
        if profile_name == "standard":
            continue  # Skip — that's just the original

        out_dir = os.path.join(output_base_dir, profile_name)
        print(f"\nApplying profile: {profile_name}")
        degrade_series(datasets, profile_name, out_dir)

    print("\nDone. Load output directories in Kevin's viewer to compare.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python degradation_engine.py <input_dicom_dir> <output_base_dir>")
        print("\nExample:")
        print("  python degradation_engine.py C:/DICOM/original C:/DICOM/degraded")
        print("\nAvailable profiles:")
        for name, params in DEGRADATION_PROFILES.items():
            print(f"  {name}: {params}")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    run_test(input_dir, output_dir)
