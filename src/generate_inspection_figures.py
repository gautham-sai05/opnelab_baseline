import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
from io import BytesIO

# Paths
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
people_mat_path = os.path.join(root, 'People.mat')
vehicle_dir = os.path.join(root, 'vehicle')
fig_dir = os.path.join(root, 'results', 'figures', 'dataset')
os.makedirs(fig_dir, exist_ok=True)

# 1. FootprintID example trace
mat = sio.loadmat(people_mat_path)
people = mat.get('people')
# MAT file stores a cell array; extract first person, first sensor, first speed
if people is not None:
    # MATLAB cell arrays become numpy object arrays
    person0 = people[0,0]  # first person
    sensor0 = person0['Sen'][0,0]  # first sensor
    speed0 = sensor0['S'][0,0]    # first speed
    # speed0 is likely a struct with field 'T' containing traces
    traces = speed0['T'] if 'T' in speed0.dtype.fields else speed0
    trace0 = traces[0,0]  # first trace (array)
    # Plot raw trace (original sampling rate 25600 Hz)
    plt.figure(figsize=(6,3))
    plt.plot(trace0)
    plt.title('FootprintID - First raw trace (25.6 kHz)')
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'footprintid_raw_trace.png'))
    plt.close()

# 2. M3N-VC example geophone waveform
# Use first zip and first geo parquet file
first_zip = os.path.join(vehicle_dir, 'a06.zip')
with zipfile.ZipFile(first_zip, 'r') as zf:
    geo_files = [f for f in zf.namelist() if f.endswith('_geo.parquet')]
    if geo_files:
        geo_path = geo_files[0]
        with zf.open(geo_path) as f:
            df = pd.read_parquet(BytesIO(f.read()))
            # Take first 1000 samples for illustration
            samples = df['samples'].values[:1000]
            plt.figure(figsize=(6,3))
            plt.plot(samples)
            plt.title('M3N-VC - Example geophone segment (1 kHz)')
            plt.xlabel('Sample index')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'm3nvc_geo_example.png'))
            plt.close()

print('Figures saved to', fig_dir)
