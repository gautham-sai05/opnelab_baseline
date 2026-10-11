import scipy.io as sio
import numpy as np, os, json

mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"

# Load the .mat file (MATLAB v7.3 may need h5py) using scipy.io.loadmat (works for v7.2)
try:
    data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
except Exception as e:
    # If it's a v7.3 file, fallback to h5py
    import h5py
    with h5py.File(mat_path, 'r') as f:
        # Convert to dict recursively
        def h5py_to_dict(g):
            out = {}
            for k, v in g.items():
                if isinstance(v, h5py.Group):
                    out[k] = h5py_to_dict(v)
                else:
                    out[k] = np.array(v)
            return out
        data = h5py_to_dict(f)

# The MATLAB file is expected to contain a struct array 'People' (or similar)
people = data.get('People')
if people is None:
    # maybe the variable is named differently (e.g., 'people')
    people = data.get('people')

if people is None:
    raise RuntimeError('Cannot locate People variable in .mat file')

# 'people' may be an array of structs; ensure it's iterable
if isinstance(people, np.ndarray):
    people_list = people.tolist()
else:
    people_list = [people]

num_people = len(people_list)
# Assume each person struct has fields: sensor, speed, amp, data (cell array of traces)
# We'll collect unique sensors, speeds, amplifications, and trace stats.
sensors = set()
speeds = set()
amplifications = set()
trace_lengths = []
nan_counts = []
inf_counts = []

for p in people_list:
    # Access fields defensively
    sensor = getattr(p, 'sensor', None) or getattr(p, 'sensor_id', None)
    speed = getattr(p, 'speed', None) or getattr(p, 'walking_speed', None)
    amp = getattr(p, 'amplification', None) or getattr(p, 'amp', None)
    sensors.add(str(sensor))
    speeds.add(str(speed))
    amplifications.add(str(amp))
    # Data field may be a cell array of numpy arrays (list of arrays)
    traces = getattr(p, 'data', None) or getattr(p, 'traces', None)
    if traces is None:
        continue
    # If it's a list or ndarray of objects, iterate
    if isinstance(traces, (list, tuple, np.ndarray)):
        for tr in traces:
            # Ensure it's a numpy array
            arr = np.array(tr, dtype=float)
            trace_lengths.append(arr.shape[0])
            nan_counts.append(int(np.isnan(arr).sum()))
            inf_counts.append(int(np.isinf(arr).sum()))
    else:
        arr = np.array(traces, dtype=float)
        trace_lengths.append(arr.shape[0])
        nan_counts.append(int(np.isnan(arr).sum()))
        inf_counts.append(int(np.isinf(arr).sum()))

# Statistics
import statistics as st
if trace_lengths:
    length_stats = {
        'min': int(min(trace_lengths)),
        'max': int(max(trace_lengths)),
        'mean': float(np.mean(trace_lengths)),
        'std': float(np.std(trace_lengths, ddof=1))
    }
else:
    length_stats = {}

report = {
    'num_people': num_people,
    'unique_sensors': sorted(list(sensors)),
    'unique_speeds': sorted(list(speeds)),
    'unique_amplifications': sorted(list(amplifications)),
    'num_traces': len(trace_lengths),
    'trace_length_stats': length_stats,
    'total_nan': int(sum(nan_counts)),
    'total_inf': int(sum(inf_counts))
}

# Write a human‑readable summary
out_path = r"c:/Users/2005g/Downloads/4691144/geophone-lmu/results/tables/footprintid_summary.txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2)
print('Report written to', out_path)
print('Summary:\n', json.dumps(report, indent=2))
