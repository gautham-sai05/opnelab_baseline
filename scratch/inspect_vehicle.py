"""
Phase 4: Validate vehicle dataset from M3N-VC _geo.parquet files.
Inspect actual files, columns, sampling rates, signal content.
"""
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

VEHICLE_DIR = r'C:\Users\2005g\Downloads\4691144\vehicle'
OUT_DIR = r'C:\Users\2005g\Downloads\4691144\geophone-lmu\data\raw\vehicle'
os.makedirs(OUT_DIR, exist_ok=True)

zip_files = [f for f in os.listdir(VEHICLE_DIR) if f.endswith('.zip')]
print(f"Found {len(zip_files)} ZIP files: {zip_files}")

all_records = []

for zip_name in sorted(zip_files):
    zip_path = os.path.join(VEHICLE_DIR, zip_name)
    run_id = os.path.splitext(zip_name)[0]
    print(f"\n=== {zip_name} (run: {run_id}) ===")
    print(f"  Size: {os.path.getsize(zip_path) / 1e6:.1f} MB")
    
    extract_dir = os.path.join(OUT_DIR, run_id)
    if not os.path.isdir(extract_dir):
        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    
    # Find _geo.parquet files
    geo_files = []
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('_geo.parquet'):
                geo_files.append(os.path.join(root, fname))
    
    print(f"  Found {len(geo_files)} _geo.parquet files")
    
    for gf in geo_files:
        try:
            df = pd.read_parquet(gf)
            print(f"\n  File: {os.path.basename(gf)}")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Shape: {df.shape}")
            print(f"    Dtypes: {dict(df.dtypes)}")
            
            if 'samples' in df.columns:
                arr = df['samples'].values.astype(float)
                print(f"    'samples' column: {len(arr)} values")
                print(f"    Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
                
                # Check for metadata columns
                for col in df.columns:
                    if col != 'samples':
                        unique = df[col].unique()
                        print(f"    '{col}': {len(unique)} unique values, first: {unique[:3]}")
                
                # Determine sampling rate from metadata or file structure
                # Check if there's a timestamp column
                if 'timestamp' in df.columns or 'time' in df.columns:
                    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
                    times = df[time_col].values
                    if len(times) > 1:
                        dt = np.diff(times[:100].astype(float))
                        sr_est = 1.0 / np.median(dt) if np.median(dt) > 0 else 'unknown'
                        print(f"    Estimated SR from timestamps: {sr_est}")
                
                all_records.append({
                    'zip': zip_name,
                    'run_id': run_id,
                    'file': os.path.basename(gf),
                    'full_path': gf,
                    'num_samples': int(len(arr)),
                    'columns': list(df.columns),
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                })
        except Exception as e:
            print(f"    ERROR reading {gf}: {e}")

print(f"\n{'='*60}")
print(f"VEHICLE DATA SUMMARY")
print(f"{'='*60}")
print(f"Total recordings: {len(all_records)}")
if all_records:
    print(f"Unique runs: {len(set(r['run_id'] for r in all_records))}")
    print(f"Total samples across all: {sum(r['num_samples'] for r in all_records)}")
    print(f"Sample counts: {[r['num_samples'] for r in all_records]}")

    with open(os.path.join(OUT_DIR, 'vehicle_metadata.json'), 'w') as f:
        json.dump(all_records, f, indent=2)
    
    # Plot first recording
    first = all_records[0]
    df = pd.read_parquet(first['full_path'])
    arr = df['samples'].values.astype(float)[:50000]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(arr, linewidth=0.3)
    ax.set_title(f"Vehicle: {first['run_id']} - {first['file']} ({first['num_samples']} samples)")
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'example_vehicle_waveform.png'), dpi=150)
    plt.close()
    print(f"\nSaved to {OUT_DIR}")
