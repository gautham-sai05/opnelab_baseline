import os, json, zipfile, pathlib, pandas as pd, numpy as np

# ---------- FootprintID audit (MATLAB v7.3) ----------
mat_path = r'c:/Users/2005g/Downloads/4691144/People.mat'
import h5py

with h5py.File(mat_path, 'r') as f:
    # Top‑level keys
    top_keys = list(f.keys())
    # Usually a struct array stored as a dataset named 'People'
    people_name = None
    for k in top_keys:
        if 'People' in k:
            people_name = k
            break
    if people_name is None:
        raise RuntimeError('Could not locate People variable in .mat file')
    people_grp = f[people_name]
    # The struct array is stored as a group with fields as datasets of size (1, N)
    # Determine number of persons from the first field length
    first_field = list(people_grp.keys())[0]
    n_people = people_grp[first_field].shape[0]
    # Containers for metadata
    sensors = set()
    speeds = set()
    amps = set()
    trace_lengths = []
    nan_counts = 0
    inf_counts = 0
    total_traces = 0
    # Loop over each person (MATLAB stores column‑major, so we index along axis 0)
    for i in range(n_people):
        # Each field is a dataset of shape (1, N) where the element is a reference
        # to the actual data stored elsewhere. We follow the reference.
        # Helper to read a scalar string field
        def read_scalar(field):
            dset = people_grp[field][:, i]
            # h5py returns a bytes object for MATLAB char arrays
            if isinstance(dset, np.ndarray):
                # flatten and decode bytes
                return bytes(dset.tobytes()).decode('utf-8', errors='ignore').strip('\x00')
            return str(dset)
        # Common field names (observed from the original repository's read_data.m)
        # The exact naming may differ; we attempt a few possibilities.
        possible_fields = ['sensor', 'sensor_id', 'sensorName']
        sensor_val = None
        for fld in possible_fields:
            if fld in people_grp:
                sensor_val = read_scalar(fld)
                break
        if sensor_val:
            sensors.add(sensor_val)
        possible_speeds = ['speed', 'walking_speed']
        speed_val = None
        for fld in possible_speeds:
            if fld in people_grp:
                speed_val = read_scalar(fld)
                break
        if speed_val:
            speeds.add(speed_val)
        possible_amps = ['amplification', 'amp']
        amp_val = None
        for fld in possible_amps:
            if fld in people_grp:
                amp_val = read_scalar(fld)
                break
        if amp_val:
            amps.add(amp_val)
        # The actual trace data are stored as a cell array under a field called
        # "data" or "traces" which contains references to numeric arrays.
        trace_field = None
        for fld in ['data', 'traces']:
            if fld in people_grp:
                trace_field = fld
                break
        if not trace_field:
            continue
        # Each entry is a reference (object) to a separate dataset
        # The reference IDs are stored as uint64 numbers; we resolve them.
        refs = people_grp[trace_field][:, i]
        for ref in refs:
            # Resolve the object reference
            obj = f[ref]
            arr = np.array(obj)
            total_traces += 1
            trace_lengths.append(arr.shape[0])
            nan_counts += int(np.isnan(arr).sum())
            inf_counts += int(np.isinf(arr).sum())

# Compute length stats
if trace_lengths:
    length_stats = {
        'min': int(min(trace_lengths)),
        'max': int(max(trace_lengths)),
        'mean': float(np.mean(trace_lengths)),
        'std': float(np.std(trace_lengths, ddof=1))
    }
else:
    length_stats = {}

footprint_report = {
    'num_people': int(n_people),
    'unique_sensors': sorted(list(sensors)),
    'unique_speeds': sorted(list(speeds)),
    'unique_amplifications': sorted(list(amps)),
    'num_traces': total_traces,
    'trace_length_stats': length_stats,
    'total_nan': int(nan_counts),
    'total_inf': int(inf_counts)
}

out_dir = r'c:/Users/2005g/Downloads/4691144/geophone-lmu/results/tables'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'footprintid_summary.json'), 'w') as f:
    json.dump(footprint_report, f, indent=2)
print('FootprintID report written to', out_dir)

# ---------- Vehicle (M3N‑VC) audit ----------
vehicle_root = r'c:/Users/2005g/Downloads/4691144/vehicle'
extract_root = r'c:/Users/2005g/Downloads/4691144/geophone-lmu/results/tmp/vehicle_audit'
os.makedirs(extract_root, exist_ok=True)
scene_reports = []

for zip_name in os.listdir(vehicle_root):
    if not zip_name.lower().endswith('.zip'):
        continue
    scene = zip_name.replace('.zip', '')
    scene_dir = os.path.join(extract_root, scene)
    os.makedirs(scene_dir, exist_ok=True)
    zip_path = os.path.join(vehicle_root, zip_name)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(scene_dir)
    # Walk the extracted folder for parquet files
    parquet_files = []
    for root, _, files in os.walk(scene_dir):
        for fname in files:
            if fname.lower().endswith('.parquet'):
                parquet_files.append(os.path.join(root, fname))
    # Identify run_ids file and geophone files
    run_ids_path = None
    geo_paths = []
    for pf in parquet_files:
        if os.path.basename(pf) == 'run_ids.parquet':
            run_ids_path = pf
        elif pf.lower().endswith('_geo.parquet'):
            geo_paths.append(pf)
    # Load run_ids if present
    runs = []
    labels = set()
    if run_ids_path:
        df_runs = pd.read_parquet(run_ids_path)
        runs = df_runs.to_dict('records')
        labels.update(df_runs['label'].unique().tolist())
    # Compute duration stats from the geophone files (they contain a "samples" column)
    durations = []
    sample_counts = []
    timestamps = []
    for gp in geo_paths:
        df = pd.read_parquet(gp)
        # Expect columns: timestamp, channel, samples
        if 'samples' in df.columns:
            sample_counts.append(int(df['samples'].sum()))
        if 'timestamp' in df.columns:
            ts = df['timestamp'].values
            timestamps.append(ts)
            # Approximate duration using first and last timestamps (seconds)
            if ts.size > 0:
                durations.append(float(ts[-1] - ts[0]))
    # Estimate sampling rate from timestamps if possible (average delta)
    sampling_rates = []
    for ts in timestamps:
        if ts.size > 1:
            deltas = np.diff(ts)
            # Convert to seconds if timestamps are in seconds; we assume they are.
            mean_dt = np.mean(deltas)
            if mean_dt > 0:
                sampling_rates.append(1.0 / mean_dt)
    scene_report = {
        'scene': scene,
        'num_geophone_files': len(geo_paths),
        'num_runs': len(runs),
        'run_labels': sorted(list(labels)),
        'total_samples': int(sum(sample_counts)),
        'duration_stats_seconds': {
            'min': float(min(durations)) if durations else None,
            'max': float(max(durations)) if durations else None,
            'mean': float(np.mean(durations)) if durations else None,
            'std': float(np.std(durations, ddof=1)) if durations else None,
        },
        'estimated_sampling_rate_hz': float(np.mean(sampling_rates)) if sampling_rates else None,
        'missing_run_ids': run_ids_path is None,
        'missing_geo_files': len(geo_paths) == 0,
    }
    scene_reports.append(scene_report)

vehicle_report_path = os.path.join(out_dir, 'vehicle_summary.json')
with open(vehicle_report_path, 'w') as f:
    json.dump(scene_reports, f, indent=2)
print('Vehicle report written to', vehicle_report_path)
