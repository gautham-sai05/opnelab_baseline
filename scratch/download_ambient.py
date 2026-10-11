# download_ambient.py
"""
Download a short (~2 min) continuous vertical component (BHZ) from a verified IRIS/FDSN station.
We use ObsPy's client to request data, convert to a NumPy array, and save raw waveform and provenance.
The script saves:
  data/raw/ambient/ambient_waveform.npy
  data/raw/ambient/ambient_metadata.json
and a quick plot under results/figures/dataset/ambient_waveform_example.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Parameters – choose a well‑known GSN station (IU ANMO) and a recent 2‑minute window
NETWORK = "IU"
STATION = "ANMO"
LOCATION = "00"  # standard location code (may also be "--")
CHANNEL = "BHZ"
STARTTIME = UTCDateTime("2022-01-01T00:00:00")  # adjust if needed
DURATION_SECONDS = 120  # 2 minutes
ENDTIME = STARTTIME + DURATION_SECONDS

OUT_DIR = r"C:\Users\2005g\Downloads\4691144\geophone-lmu\data\raw\ambient"
os.makedirs(OUT_DIR, exist_ok=True)

client = Client("EARTHSCOPE")
print(f"Requesting {NETWORK} {STATION} {LOCATION} {CHANNEL} {STARTTIME} – {ENDTIME}")
st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL, STARTTIME, ENDTIME)
# Merge all traces in the stream to ensure continuity and fill gaps
st.merge(method=1, fill_value='interpolate')
tr = st[0]
data = tr.data.astype(np.float64)
# Preserve original sampling rate and timestamps
orig_sr = tr.stats.sampling_rate
start_time_iso = str(tr.stats.starttime)
end_time_iso = str(tr.stats.endtime)

# Save raw waveform
wave_path = os.path.join(OUT_DIR, "ambient_waveform.npy")
np.save(wave_path, data)

# Save provenance metadata
meta = {
    "network": NETWORK,
    "station": STATION,
    "location": LOCATION,
    "channel": CHANNEL,
    "start_time": start_time_iso,
    "end_time": end_time_iso,
    "sampling_rate": orig_sr,
    "samples": int(len(data)),
    "source_url": "https://service.earthscope.org/fdsnws/dataselect/1/query",
    "license": "Data provided by IRIS/DMC under CC‑0 (public domain)",
    "note": "Vertical component broadband seismometer, not a geophone. Used as ambient background class only."
}
meta_path = os.path.join(OUT_DIR, "ambient_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# Plot first 10 seconds for quick visual check
plt.figure(figsize=(12, 4))
sec_to_plot = 10
samples_to_plot = int(orig_sr * sec_to_plot)
plt.plot(np.arange(samples_to_plot) / orig_sr, data[:samples_to_plot])
plt.title(f"Ambient {STATION} {CHANNEL} – first {sec_to_plot}s")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "ambient_waveform_example.png")
plt.savefig(plot_path, dpi=150)
plt.close()

print("Ambient download and save completed.")
print(f"Waveform shape: {data.shape}, sampling rate: {orig_sr} Hz")
print(f"Saved waveform: {wave_path}")
print(f"Saved metadata: {meta_path}")
print(f"Saved plot: {plot_path}")
