"""
Generate sample CT dose data for testing LeapfrogDose
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_exams=500, output_file="sample_dose_data.csv"):
    """
    Generate realistic sample CT dose data for testing.
    
    Includes:
    - Adult and pediatric exams
    - Multiple body regions
    - Routine and non-routine protocols
    - Multi-phase studies
    - Outliers
    """
    
    data = []
    
    # Define protocols with realistic DLP distributions
    protocols = {
        # Adult protocols (region, mean_dlp, std_dlp, is_routine)
        "Head CT": ("Head", 950, 150, True),
        "Routine Head": ("Head", 920, 140, True),
        "CTA Head": ("Head", 1800, 300, False),
        "Head CT 3 Phase": ("Head", 2200, 400, False),
        
        "Chest CT": ("Chest", 350, 100, True),
        "Routine Chest": ("Chest", 330, 90, True),
        "CTA Chest": ("Chest", 800, 200, False),
        "Chest CT Pre+Post": ("Chest", 1100, 250, False),
        
        "Abd/Pelvis CT": ("Abdomen-Pelvis", 650, 150, True),
        "Routine Abdomen": ("Abdomen-Pelvis", 620, 140, True),
        "Abd/Pelvis 3 Phase": ("Abdomen-Pelvis", 1600, 350, False),
        "CTA Abdomen": ("Abdomen-Pelvis", 1200, 280, False),
        
        "Chest Abd Pelvis": ("Chest-Abdomen-Pelvis", 850, 200, True),
        "CAP CT": ("Chest-Abdomen-Pelvis", 820, 190, True),
        
        "Spine CT": ("Spine", 680, 160, True),
        "Lumbar Spine": ("Spine", 650, 150, True),
        
        # Pediatric protocols
        "Head CT Peds": ("Head", 350, 80, True),
        "Chest CT Peds": ("Chest", 150, 40, True),
        "Abd/Pelvis Peds": ("Abdomen-Pelvis", 200, 60, True),
    }
    
    # Age distributions
    adult_ages = list(range(18, 90))
    peds_ages = [0.5, 0.8, 2, 3, 4, 6, 7, 8, 11, 12, 13, 16, 17]
    
    # Scanners
    scanners = ["CT Scanner 1", "CT Scanner 2", "CT Scanner 3"]
    
    # Generate exams
    start_date = datetime(2024, 1, 1)
    
    for i in range(n_exams):
        # Select protocol
        protocol_name = np.random.choice(list(protocols.keys()))
        region, mean_dlp, std_dlp, is_routine = protocols[protocol_name]
        
        # Determine age
        if "Peds" in protocol_name:
            age = np.random.choice(peds_ages)
        else:
            age = np.random.choice(adult_ages)
        
        # Generate DLP (with some outliers)
        if np.random.random() < 0.05:  # 5% outliers
            dlp = mean_dlp + std_dlp * np.random.uniform(3, 6)
        else:
            dlp = max(10, np.random.normal(mean_dlp, std_dlp))
        
        # Generate CTDIvol (rough approximation)
        ctdivol = dlp / np.random.uniform(12, 20)
        
        # Generate exam date
        days_offset = np.random.randint(0, 365)
        exam_date = start_date + timedelta(days=days_offset)
        
        # Scanner
        scanner = np.random.choice(scanners)
        
        # Patient ID
        patient_id = f"PT{i+1:05d}"
        
        data.append({
            "Patient ID": patient_id,
            "Age": age,
            "Study Description": protocol_name,
            "Exam Date": exam_date.strftime("%Y-%m-%d"),
            "DLP": round(dlp, 1),
            "CTDI vol": round(ctdivol, 2),
            "Scanner": scanner,
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated {n_exams} sample CT dose records")
    print(f"Saved to: {output_file}")
    print(f"\nBreakdown:")
    print(f"  Adult exams: {len(df[df['Age'] >= 18])}")
    print(f"  Pediatric exams: {len(df[df['Age'] < 18])}")
    print(f"  Protocols: {df['Study Description'].nunique()}")
    print(f"  Date range: {df['Exam Date'].min()} to {df['Exam Date'].max()}")
    print(f"\nNow run:")
    print(f"  python leapfrog_dose.py {output_file} 'Demo Medical Center'")
    
    return df


if __name__ == "__main__":
    generate_sample_data()