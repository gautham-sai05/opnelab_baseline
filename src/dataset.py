import numpy as np
import scipy.io as sio
import os
import zipfile
import pandas as pd
from .ambient import load_ambient_data
from .preprocessing import preprocess_waveform, create_windows, standardize_windows
from .synthetic import generate_synthetic_traces

def load_people_mat(filepath):
    if not os.path.exists(filepath):
        print(f'People.mat not found at {filepath}')
        return []
    mat = sio.loadmat(filepath)
    key = 'people' if 'people' in mat else 'People'
    if key not in mat:
        print('No people key found in mat.')
        return []
    p_cell = mat[key]
    traces = []
    for person_idx in range(p_cell.shape[1]):
        person = p_cell[0, person_idx]
        sen_obj = person['Sen'][0, 0]
        flat = sen_obj.flatten()
        for rec_idx, cell in enumerate(flat):
            if not isinstance(cell, np.ndarray):
                continue
            try:
                arr = np.array(cell, dtype=float)
            except Exception:
                continue
            if arr.ndim == 2:
                if arr.shape[1] > 0:
                    arr = arr[:, 0]
                else:
                    continue
            if arr.ndim != 1 or len(arr) < 1000:
                continue
            traces.append({
                'data': arr,
                'subject': f'human_{person_idx}',
                'label': 'human',
                'sr': 25600,  # original sampling rate from People.mat (25.6 kHz)
                'recording_id': f'human_{person_idx}_{rec_idx}',
            })
    print(f'Loaded {len(traces)} human traces from People.mat.')
    return traces

def get_vehicle_and_noise_data(vehicle_zip_dir):
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'vehicle')
    os.makedirs(temp_dir, exist_ok=True)
    traces = []
    if not os.path.isdir(vehicle_zip_dir):
        print(f'Vehicle directory not found at {vehicle_zip_dir}')
        return _mock_noise()
    for zip_name in os.listdir(vehicle_zip_dir):
        if not zip_name.lower().endswith('.zip'):
            continue
        zip_path = os.path.join(vehicle_zip_dir, zip_name)
        extract_path = os.path.join(temp_dir, os.path.splitext(zip_name)[0])
        if not os.path.isdir(extract_path):
            print(f'Extracting {zip_name}...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_path)
        rec_id_base = os.path.splitext(zip_name)[0]
        rec_counter = 0
        for root, _, files in os.walk(extract_path):
            for fname in files:
                if not fname.lower().endswith('_geo.parquet'):
                    continue
                full_path = os.path.join(root, fname)
                try:
                    df = pd.read_parquet(full_path)
                    if 'samples' in df.columns:
                        arr = df['samples'].values.astype(float)
                        arr = arr[:100000] # Memory cap
                        # Compute original sampling rate from timestamp column if available
                        if 'timestamp' in df.columns:
                            diffs = df['timestamp'].diff().dropna()
                            median_dt = diffs.median()
                            orig_sr = int(round(1e3 / median_dt)) if median_dt != 0 else 1000
                        else:
                            orig_sr = 1000
                        traces.append({
                            'data': arr,
                            'subject': f'vehicle_{rec_id_base}',
                            'label': 'vehicle',
                            'sr': orig_sr,
                            'recording_id': f'vehicle_{rec_id_base}_{rec_counter}',
                        })
                        rec_counter += 1
                except Exception as e:
                    print(f'  Failed to load {full_path}: {e}')
    print(f'Loaded {len(traces)} vehicle traces.')
    traces += _mock_noise()
    return traces

def _mock_noise():
    traces = []
    for i in range(50):
        noise = np.random.randn(10000) * 0.05
        traces.append({
            'data': noise,
            'subject': f'noise_{i}',
            'label': 'noise',
            'sr': 1000,  # synthetic ambient kept at 1kHz for compatibility
            'recording_id': f'noise_{i}'
        })
    return traces

def prepare_dataset(config):
    human_source = config['dataset'].get('human_source',
                       r'C:\Users\2005g\Downloads\4691144\People.mat')
    vehicle_zip_dir = config['dataset'].get('vehicle_source',
                          r'C:\Users\2005g\Downloads\4691144\vehicle')
    human_traces = load_people_mat(human_source)
    veh_noise_traces = get_vehicle_and_noise_data(vehicle_zip_dir)
    ambient_traces = load_ambient_data()
    all_traces = human_traces + veh_noise_traces + ambient_traces
    sr_target = config['dataset']['sampling_rate']
    synthetic_config = config['dataset'].get('synthetic', {})
    if synthetic_config.get('enabled', False):
        synthetic_traces = generate_synthetic_traces(
            traces_per_class=int(synthetic_config.get('traces_per_class', 30)),
            samples_per_trace=int(synthetic_config.get('samples_per_trace', 10000)),
            sampling_rate=sr_target,
            seed=int(synthetic_config.get('seed', 42)),
            label_noise=float(synthetic_config.get('label_noise', 0.0)),
        )
        if synthetic_config.get('replace_real', False):
            all_traces = synthetic_traces
        else:
            all_traces += synthetic_traces
    label_map = {'noise': 0, 'human': 1, 'vehicle': 2}
    win_samples = config['dataset']['window_samples']
    hop_samples = config['dataset']['hop_samples']
    X, y, metadata = [], [], []
    for trace in all_traces:
        data = preprocess_waveform(trace['data'], orig_sr=trace['sr'], target_sr=sr_target)
        windows = create_windows(data, win_samples, hop_samples)
        lbl = label_map[trace['label']]
        for w_idx, w in enumerate(windows):
            X.append(w)
            y.append(lbl)
            metadata.append({
                'recording_id': trace['recording_id'],
                'subject_id': trace['subject'],
                'class': trace['label'],
                'label': lbl,
            })
    X = np.array(X, dtype=np.float32)
    X = standardize_windows(X)
    y = np.array(y, dtype=np.int64)
    df = pd.DataFrame(metadata)
    if synthetic_config.get('enabled', False) and synthetic_config.get('replace_real', False):
        processed_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        )
        os.makedirs(processed_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(processed_dir, 'synthetic_dataset.npz'),
            X=X,
            y=y,
        )
    rng = np.random.default_rng(42)
    train_subs, val_subs, test_subs = set(), set(), set()
    for _, class_rows in df.groupby('class', sort=True):
        subjects = np.array(class_rows['subject_id'].unique(), dtype=object)
        rng.shuffle(subjects)
        n = len(subjects)
        train_end = max(1, int(0.70 * n))
        val_end = max(train_end + 1, int(0.85 * n)) if n > 2 else train_end
        train_subs.update(subjects[:train_end])
        val_subs.update(subjects[train_end:val_end])
        test_subs.update(subjects[val_end:])
    df['split'] = df['subject_id'].apply(
        lambda s: 'train' if s in train_subs else ('val' if s in val_subs else 'test'))
    split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'splits'))
    os.makedirs(split_dir, exist_ok=True)
    for s in ('train', 'val', 'test'):
        df[df['split'] == s].to_csv(os.path.join(split_dir, f'{s}.csv'), index=False)
    def subset(split):
        idx = df.index[df['split'] == split].tolist()
        return X[idx], y[idx]
    X_train, y_train = subset('train')
    X_val,   y_val   = subset('val')
    X_test,  y_test  = subset('test')
    print(f'Dataset ready: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}')
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), df
