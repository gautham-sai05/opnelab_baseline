import os
import json
import numpy as np


def load_ambient_data():
    """Load ambient waveform and metadata.

    Expects files under ``data/raw/ambient``:
        - ``ambient_waveform.npy`` : NumPy array of the raw signal
        - ``ambient_metadata.json`` : JSON with sampling rate and provenance

    Returns a list of trace dictionaries compatible with the rest of the pipeline.
    If the files are missing, returns an empty list and prints a warning.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ambient'))
    waveform_path = os.path.join(base_dir, 'ambient_waveform.npy')
    metadata_path = os.path.join(base_dir, 'ambient_metadata.json')

    if not os.path.exists(waveform_path) or not os.path.exists(metadata_path):
        print('Ambient data not found; skipping ambient class.')
        return []

    data = np.load(waveform_path).astype(float)
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    sr = meta.get('sampling_rate', 1000)
    rec_id = meta.get('station', 'ambient') + '_' + meta.get('channel', 'BHZ')
    trace = {
        'data': data,
        'subject': rec_id,
        'label': 'noise',
        'sr': int(sr),
        'recording_id': f'ambient_{rec_id}'
    }
    return [trace]
