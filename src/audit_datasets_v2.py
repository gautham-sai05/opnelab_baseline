import os, json, zipfile, numpy as np, pandas as pd, scipy.io as sio, pyarrow as pa

# ---------------------------------------------------------------------
# 1. FootprintID (People.mat) – load using scipy.io (MATLAB v7.2 compatible)
# ---------------------------------------------------------------------
mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"
footprint_report = {}
try:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    # Find the primary variable that is not a meta entry
    primary_key = next(k for k in mat.keys() if not k.startswith('__'))
    people = mat[primary_key]
    # 'people' is expected to be a numpy object array (length 10)
    if isinstance(people, np.ndarray):
        people_list = people.tolist()
    else:
        people_list = [people]
    num_people = len(people_list)
    sensors = set()
    speeds = set()
    amps = set()
    trace_lengths = []
    nan_total = 0
    inf_total = 0
    num_traces = 0
    for person in people_list:
        # fields may be attributes on a simple namespace object
        for field in ['sensor', 'sensor_id', 'sensorName']:
            if hasattr(person, field):
                sensors.add(str(getattr(person, field)))
                break
        for field in ['speed', 'walking_speed']:
            if hasattr(person, field):
                speeds.add(str(getattr(person, field)))
                break
        for field in ['amplification', 'amp']:
            if hasattr(person, field):
                amps.add(str(getattr(person, field)))
                break
        # waveform data stored in a field called 'data' or 'traces'
        traces = None
        for field in ['data', 'traces']:
            if hasattr(person, field):
                traces = getattr(person, field)
                break
        if traces is None:
            continue
        # traces is a cell array – represented as a list or ndarray of arrays
        if isinstance(traces, np.ndarray):
            iter_tr = traces.tolist()
        else:
            iter_tr = list(traces)
        for tr in iter_tr:
            arr = np.asarray(tr, dtype=float)
            num_traces += 1
            trace_lengths.append(arr.shape[0])
            nan_total += int(np.isnan(arr).sum())
            inf_total += int(np.isinf(arr).sum())
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
        'num_people': int(num_people),
        'unique_sensors': sorted(list(sensors)),
        'unique_speeds': sorted(list(speeds)),
        'unique_amplifications': sorted(list(amps)),
        'num_traces': int(num_traces),
        'trace_length_stats': length_stats,
        'total_nan': int(nan_total),
        'total_inf': int(inf_total)
    }
except Exception as e:
    footprint_report = {'error': str(e)}

# ---------------------------------------------------------------------
# 2. Vehicle (M3N‑VC) – inspect each zip without extracting
# ---------------------------------------------------------------------
vehicle_dir = r"c:/Users/2005g/Downloads/4691144/vehicle"
vehicle_report = []
for zip_name in sorted(os.listdir(vehicle_dir)):
    if not zip_name.lower().endswith('.zip'):
        continue
    scene = zip_name.replace('.zip', '')
    zip_path = os.path.join(vehicle_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        # Identify parquet files
        parquet_files = [n for n in names if n.lower().endswith('.parquet')]
        geo_files = [n for n in parquet_files if n.lower().endswith('_geo.parquet')]
        run_ids_files = [n for n in parquet_files if os.path.basename(n) == 'run_ids.parquet']
        # Load run_ids if present
        labels = []
        durations = []
        if run_ids_files:
            with zf.open(run_ids_files[0]) as f:
                try:
                    df_runs = pd.read_parquet(f)
                except Exception:
                    # Fallback to pyarrow directly
                    table = pa.parquet.read_table(f)
                    df_runs = table.to_pandas()
            if 'label' in df_runs.columns:
                labels = df_runs['label'].astype(str).unique().tolist()
            # Duration may be given in a 'length' column (seconds)
            if 'length' in df_runs.columns:
                durations = df_runs['length'].astype(float).tolist()
        # Compute total samples from geo files
        total_samples = 0
        for gf in geo_files:
            with zf.open(gf) as f:
                try:
                    df_geo = pd.read_parquet(f)
                except Exception:
                    table = pa.parquet.read_table(f)
                    df_geo = table.to_pandas()
            if 'samples' in df_geo.columns:
                total_samples += int(df_geo['samples'].sum())
        scene_info = {
            'scene': scene,
            'num_parquet_files': len(parquet_files),
            'num_geo_parquet_files': len(geo_files),
            'has_run_ids': bool(run_ids_files),
            'run_labels': sorted(labels),
            'total_samples': total_samples,
            'duration_stats_seconds': {
                'min': float(min(durations)) if durations else None,
                'max': float(max(durations)) if durations else None,
                'mean': float(np.mean(durations)) if durations else None,
                'std': float(np.std(durations, ddof=1)) if durations else None,
            }
        }
        vehicle_report.append(scene_info)

# ---------------------------------------------------------------------
# 3. Write JSON reports
# ---------------------------------------------------------------------
out_dir = r"c:/Users/2005g/Downloads/4691144/geophone-lmu/results/tables"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'footprintid_summary.json'), 'w') as f:
    json.dump(footprint_report, f, indent=2)
with open(os.path.join(out_dir, 'vehicle_summary.json'), 'w') as f:
    json.dump(vehicle_report, f, indent=2)
print('Reports written to', out_dir)
