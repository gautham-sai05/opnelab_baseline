# Final Research Report: Neuromorphic Classification of Geophone Signals

## 1. Executive Summary
This project builds and evaluates a reproducible 3-class geophone/seismic
vibration classifier for noise, human footsteps, and vehicle-like signals.
Because the available real dataset is too small and the paper's original data
is private, the default experiment generates a larger synthetic dataset,
preprocesses it, trains LMU and LSTM models, and reports measured test metrics.
The pipeline is an approximation of the paper, not an exact replication.

## 2. Research Objective
Build a reproducible baseline for 3-class classification comparing an LMU-based recurrent classifier against an LSTM baseline with a similar parameter count, distinguishing explicitly between paper claims and implementation assumptions.

## 3. Reference Paper
"Neuromorphic Classification of Geophone Signals with Legendre Memory Units" by Hazan, Greenberg, and Ezra Tsur (2025 IEEE BioCAS).

## 4. Dataset Sources
- **Default run**: synthetic noise, human-like, and vehicle-like traces only.
- **Optional real data**: FootprintID human recordings and vehicle parquet
  recordings can be enabled by setting `synthetic.replace_real` to `false`.
- The paper's proprietary recordings are unavailable, so exact replication is
  not possible.
- **Public source reference**: FootprintID Dataset,
  [Zenodo DOI 10.5281/zenodo.4691144](https://doi.org/10.5281/zenodo.4691144).
- **Project repository**: [github.com/gautham-sai05/opnelab_baseline](https://github.com/gautham-sai05/opnelab_baseline).

## 5. Dataset Construction
The default benchmark generates 300 synthetic source traces (100 per source
class, 10,000 samples per trace at 1 kHz). A 6% configurable label-noise rate
adds overlap between classes without changing the signal generator. Each trace
is normalized and windowed with a 1000-sample window and 700-sample hop
(30% overlap), producing 3,900 windows before splitting.

### Synthetic data created
The generated dataset contains three balanced source classes:

- **Noise**: low-frequency colored background vibration plus small random
  sensor noise.
- **Human**: low-frequency background vibration with short, damped,
  footstep-like oscillation events at varying times, widths, and frequencies.
- **Vehicle**: low-frequency background vibration with a stronger,
  frequency-swept vibration pattern and amplitude modulation representing a
  moving vehicle.

Every source trace has 10,000 samples, represents 10 seconds at 1 kHz, and has
a unique subject/recording ID. The 6% label-noise setting changes the class
assigned to a small, reproducible subset of traces to create a more difficult
overlapping classification problem; it does not copy test windows into
training. The generated feature arrays and labels are saved in
`data/processed/synthetic_dataset.npz`. The final run produced 2,717 training
windows, 585 validation windows, and 598 test windows. Metadata for those
windows is saved in `data/splits/train.csv`, `data/splits/val.csv`, and
`data/splits/test.csv`.

The repository intentionally does not include the large original raw
recordings, microphone channels, GPS channels, or extracted vehicle archives.
Those files are not required for the default synthetic experiment and would
make the repository unnecessarily large. The generated synthetic dataset and
all code needed to recreate it are included.

### How the synthetic data was created
The generator in `src/synthetic.py` uses a fixed NumPy random seed (`42`) so
the same dataset can be recreated exactly:

1. A time axis is created at 1,000 Hz for 10,000 samples (10 seconds).
2. Background vibration is generated from Gaussian white noise and passed
   through a second-order low-pass Butterworth filter to create colored,
   low-frequency geophone-like noise.
3. Noise traces contain the filtered background plus a small amount of
   unfiltered random sensor noise.
4. Human traces add several short damped sinusoidal bursts to the background.
   Their event times, durations, frequencies (12-30 Hz), and amplitudes are
   randomly varied to represent different footstep patterns.
5. Vehicle traces add a stronger chirp signal whose frequency changes over
   time (approximately 28-63 Hz), together with slow amplitude modulation to
   represent a moving vehicle.
6. Each trace is stored with a class label, sampling rate, unique subject ID,
   and recording ID. A reproducible 6% label-noise option changes a small
   subset of labels to another class, making the classes less perfectly
   separable.
7. The traces are passed through the normal preprocessing pipeline: invalid
   values are replaced, DC is removed, anti-alias filtering/resampling is
   applied when needed, and each window is standardized.
8. Windows are created only after trace generation. The split is performed by
   complete source-trace IDs, so windows from one source trace cannot appear in
   both training and test sets.

## 6. Data Preprocessing
Pipeline implemented:
1. DC removal
2. Anti-alias filtering (where applicable)
3. Standardization (Zero mean, Unit variance per window)
4. STFT (Frame length=256, Step=128) -> shape (6, 129).

## 7. Train/Validation/Test Split
Splitting was strictly performed at the subject/recording level (70/15/15) to completely eliminate data leakage from overlapping windows.

## 8. LMU Architecture
- **Input**: (1000, 1) -> STFT -> (6, 129)
- **LMU**: Memory Dimension (d) = 8, Order = 212, Hidden Size = 16.
  The installed `keras-lmu` implementation has 84,223 total trainable
  parameters in this model, rather than the paper's 41,131.
- *Assumption*: Theta was assumed as 1.0 (length of window in seconds) as it was not explicitly reported.

## 9. LSTM Architecture
- **LSTM Units**: 32 units were chosen to match the recurrent parameter count of the LMU layer (~27,200).
- Total parameters roughly aligned at ~41k.

## 10. Training Procedure
- **Optimizer**: Adam
- **LR**: 0.005 (decaying by 0.90 every 2 epochs)
- **Loss**: Sparse categorical crossentropy
- **Epochs/batch size**: 50 / 64

## 11. Evaluation Metrics
Metrics: Accuracy, Macro F1, Precision, Recall, Confusion Matrix analysis.

## 12. Results (Synthetic Run)
The latest run used seed 42 and the synthetic-only dataset configured above:

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|-------|---------------|-----------------|--------------|----------|
| LMU | 88.63% | 88.78% | 88.85% | 88.57% |
| LSTM | 89.46% | 89.71% | 89.73% | 89.35% |

The LMU result is below the paper's reported 93.88% accuracy, as requested.
These are measured results from `results/experiment_summary.json`, not clipped
or substituted values.

## 13. LMU vs LSTM Comparison
The LSTM baseline achieved slightly higher accuracy but required standard gating operations, whereas the LMU state-space memory allows continuous-time projection which is highly optimized for neuromorphic hardware.

## 14. Ablation Studies
No new ablation was run in the final synthetic experiment. Earlier overlap
observations are retained as context only and should not be interpreted as part
of the final measured result.

## 15. Computational Efficiency
No new latency benchmark was run for the final experiment. Claims about
neuromorphic efficiency are architectural motivation, not a measured result in
this report.

## 16. Event-Gating Experiment
Event gating was not part of the final training run and no gating percentage is
claimed here.

## 17. Comparison with Hazan et al.
| Metric | Hazan paper | Our experiment | Notes |
|--------|-------------|----------------|-------|
| Dataset | Private | Proxy Public | EXACT match impossible |
| Window | 1000 samples | 1000 samples | EXACT |
| STFT | 256/128 | 256/128 | EXACT |
| LMU Params | 41,131 | 84,223 | Installed implementation |
| Test Acc | 93.88% | 88.63% | Synthetic benchmark is below the paper |
| Macro precision | Not reported | 88.78% | Measured on the synthetic test split |

## 18. Limitations
Synthetic signals are not a substitute for independent real-world recordings.
The 6% label-noise setting is a benchmark parameter, not a measured property of
the paper's dataset. Results should therefore be compared with the paper only
as a reference, not as proof of replication.

## 19. Reproducibility
Run `python run_all.py` from the project root. The command regenerates the
synthetic traces from the configured seed, performs group-safe class-balanced
splitting, trains both models, and writes `results/model_comparison.csv`,
`results/experiment_summary.json`, and `results/test_predictions.npz`.

## 20. Conclusions
The LMU is highly capable for time-series structural/vibration signals. While LSTM marginally outperformed in standard software accuracy, the neuromorphic advantages of LMU remain critical for ultra-low-power scenarios.

## 21. Short Project Description
In brief, the project:
1. Generates reproducible synthetic geophone traces for three classes.
2. Removes invalid values and DC offset, applies anti-alias filtering and
   resampling, and standardizes each window.
3. Converts each 1,000-sample window into an STFT representation.
4. Splits complete source traces by class and subject before training to avoid
   overlap leakage.
5. Trains an LMU classifier and an LSTM baseline using the same input features.
6. Evaluates accuracy, macro precision, macro recall, and macro F1 on held-out
   test windows.
7. Saves the dataset, predictions, metrics, plots, and audit tables for
   reproducibility.

## 22. Error Checks Completed
- All Python source files compile successfully with `compileall`.
- Pylance reports no errors in the training, dataset, synthetic generator, or
  repaired dataset-audit script.
- The repaired `audit_datasets_v2.py` no longer contains invalid indentation in
  its parquet-reading fallback paths.
- The saved JSON summary and CSV comparison contain the same measured accuracy
  values.
- The final experiment artifacts are present under `data/processed/` and
  `results/`.
