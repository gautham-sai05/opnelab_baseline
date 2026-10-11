"""
Correct extraction of People.mat.
From debug: sensor_obj has dtype [('S', 'O')]  -> single field 'S' containing object array.
In MATLAB: .S{speed_scale}{trace_num, amplification}
In Python: sensor['S'] -> object array -> [0, speed_idx] -> object array -> [trace, amp] -> waveform

Full path:
  people[0, person_idx]['Sen'][0,0][0, sensor_idx]['S'][0,0] -> 
    This gives us a (1, 8) or similar object array (8 speed scales).
    Then [0, speed_idx] -> (N_traces, 3) object array.
    Then [trace, amp] -> (N_samples, 1) float64 waveform.
"""
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

PEOPLE_MAT = r'C:\Users\2005g\Downloads\4691144\People.mat'
OUT_DIR = r'C:\Users\2005g\Downloads\4691144\geophone-lmu\data\raw\human'
os.makedirs(OUT_DIR, exist_ok=True)

mat = sio.loadmat(PEOPLE_MAT, struct_as_record=True, squeeze_me=False)
people = mat['people']
fs = 25600  # From read_data.m

# First, let's understand the exact nesting for person 0, sensor 0
print("=== DEBUGGING EXACT NESTING ===")
person = people[0, 0]  # (1,1) struct with 'Sen'
sen_cell = person['Sen'][0, 0]  # (1,5) object array
sensor_struct = sen_cell[0, 0]  # (1,1) struct with 'S'
print(f"sensor_struct: shape={sensor_struct.shape}, dtype={sensor_struct.dtype}")

s_field = sensor_struct['S']
print(f"S field: shape={s_field.shape}, dtype={s_field.dtype}")

# Unwrap (1,1) if needed
while isinstance(s_field, np.ndarray) and s_field.shape == (1,1) and s_field.dtype == object:
    s_field = s_field[0, 0]
    print(f"  unwrapped to: shape={getattr(s_field, 'shape', 'N/A')}, dtype={getattr(s_field, 'dtype', 'N/A')}")

if isinstance(s_field, np.ndarray) and s_field.dtype == object:
    print(f"S cell array: shape={s_field.shape}")
    # Check first speed
    for si in range(min(s_field.shape[0], 2)):
        for sj in range(min(s_field.shape[1] if s_field.ndim > 1 else 1, 2)):
            if s_field.ndim > 1:
                speed_cell = s_field[si, sj]
            else:
                speed_cell = s_field[si]
            print(f"  Speed [{si},{sj}]: type={type(speed_cell)}, shape={getattr(speed_cell, 'shape', 'N/A')}, dtype={getattr(speed_cell, 'dtype', 'N/A')}")
            if isinstance(speed_cell, np.ndarray) and speed_cell.dtype == object:
                print(f"    Traces array: shape={speed_cell.shape}")
                for ti in range(min(speed_cell.shape[0], 2)):
                    for tj in range(min(speed_cell.shape[1] if speed_cell.ndim > 1 else 1, 2)):
                        if speed_cell.ndim > 1:
                            waveform = speed_cell[ti, tj]
                        else:
                            waveform = speed_cell[ti]
                        print(f"    Trace [{ti},{tj}]: type={type(waveform)}, shape={getattr(waveform, 'shape', 'N/A')}, dtype={getattr(waveform, 'dtype', 'N/A')}")
                        if isinstance(waveform, np.ndarray) and waveform.dtype != object:
                            flat = waveform.flatten()
                            print(f"      *** WAVEFORM: {len(flat)} samples, {len(flat)/fs:.2f}s, first 3: {flat[:3]}")
elif isinstance(s_field, np.ndarray) and s_field.dtype.names:
    print(f"S is a struct with fields: {s_field.dtype.names}")
else:
    print(f"S is: type={type(s_field)}, value={s_field}")

print("\n=== NOW EXTRACTING ALL TRACES ===")
all_traces = []
trace_metadata = []

for person_idx in range(people.shape[1]):
    person = people[0, person_idx]
    sen_cell = person['Sen'][0, 0]  # (1, N_sensors) 
    
    for sensor_idx in range(sen_cell.shape[1]):
        sensor_struct = sen_cell[0, sensor_idx]
        if not isinstance(sensor_struct, np.ndarray) or sensor_struct.size == 0:
            continue
        if sensor_struct.dtype.names is None or 'S' not in sensor_struct.dtype.names:
            continue
        
        s_field = sensor_struct['S']
        # Unwrap
        while isinstance(s_field, np.ndarray) and s_field.shape == (1,1) and s_field.dtype == object:
            s_field = s_field[0, 0]
        
        if not isinstance(s_field, np.ndarray) or s_field.dtype != object:
            continue
        
        # s_field is now a cell array of speed conditions
        flat_speeds = s_field.flatten()
        for speed_idx, speed_cell in enumerate(flat_speeds):
            if not isinstance(speed_cell, np.ndarray):
                continue
            
            if speed_cell.dtype == object:
                # (N_traces, 3) object array of waveforms
                for trace_idx in range(speed_cell.shape[0]):
                    n_amps = speed_cell.shape[1] if speed_cell.ndim > 1 else 1
                    for amp_idx in range(n_amps):
                        if speed_cell.ndim > 1:
                            waveform = speed_cell[trace_idx, amp_idx]
                        else:
                            waveform = speed_cell[trace_idx]
                        
                        if not isinstance(waveform, np.ndarray) or waveform.dtype == object:
                            continue
                        
                        arr = waveform.flatten().astype(np.float64)
                        if len(arr) < 1000:
                            continue
                        
                        trace_id = f"p{person_idx}_sen{sensor_idx}_spd{speed_idx}_t{trace_idx}_a{amp_idx}"
                        all_traces.append(arr)
                        trace_metadata.append({
                            'trace_id': trace_id,
                            'person_idx': int(person_idx),
                            'sensor_idx': int(sensor_idx),
                            'speed_scale': int(speed_idx),
                            'trace_num': int(trace_idx),
                            'amplification': int(amp_idx),
                            'num_samples': int(len(arr)),
                            'duration_s': round(float(len(arr) / fs), 4),
                            'sampling_rate': int(fs),
                        })
            elif speed_cell.dtype != object:
                arr = speed_cell.flatten().astype(np.float64)
                if len(arr) >= 1000:
                    trace_id = f"p{person_idx}_sen{sensor_idx}_spd{speed_idx}"
                    all_traces.append(arr)
                    trace_metadata.append({
                        'trace_id': trace_id,
                        'person_idx': int(person_idx),
                        'sensor_idx': int(sensor_idx),
                        'speed_scale': int(speed_idx),
                        'trace_num': 0,
                        'amplification': 0,
                        'num_samples': int(len(arr)),
                        'duration_s': round(float(len(arr) / fs), 4),
                        'sampling_rate': int(fs),
                    })

print(f"\nTotal traces extracted: {len(all_traces)}")
if len(all_traces) > 0:
    print(f"Unique persons: {len(set(m['person_idx'] for m in trace_metadata))}")
    print(f"Unique sensors: {sorted(set(m['sensor_idx'] for m in trace_metadata))}")
    print(f"Unique speeds: {sorted(set(m['speed_scale'] for m in trace_metadata))}")
    print(f"Duration range: {min(m['duration_s'] for m in trace_metadata):.2f}s - {max(m['duration_s'] for m in trace_metadata):.2f}s")
    print(f"Samples range: {min(m['num_samples'] for m in trace_metadata)} - {max(m['num_samples'] for m in trace_metadata)}")
    
    # Save
    with open(os.path.join(OUT_DIR, 'trace_metadata.json'), 'w') as f:
        json.dump(trace_metadata, f, indent=2)
    
    np.savez_compressed(os.path.join(OUT_DIR, 'human_traces.npz'),
                        **{m['trace_id']: t for m, t in zip(trace_metadata, all_traces)})
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    for i in range(min(3, len(all_traces))):
        t = np.arange(len(all_traces[i])) / fs
        axes[i].plot(t, all_traces[i], linewidth=0.3)
        axes[i].set_title(f"{trace_metadata[i]['trace_id']} ({len(all_traces[i])} samples @ {fs}Hz)")
        axes[i].set_ylabel('Voltage (V)')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'example_human_waveforms.png'), dpi=150)
    plt.close()
    print(f"\nFiles saved to {OUT_DIR}")
else:
    print("FAILED: No traces extracted")
