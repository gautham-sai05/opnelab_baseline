"""Reproducible synthetic geophone traces for development and benchmarking."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _colored_noise(rng: np.random.Generator, size: int, cutoff: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size)
    b, a = signal.butter(2, cutoff, btype="low")
    return signal.filtfilt(b, a, white)


def _human_trace(rng: np.random.Generator, size: int, sr: int) -> np.ndarray:
    t = np.arange(size) / sr
    trace = 0.35 * _colored_noise(rng, size, 0.18)
    n_events = max(2, size // (sr * 2))
    for _ in range(n_events):
        center = rng.uniform(0, t[-1])
        width = rng.uniform(0.025, 0.08)
        frequency = rng.uniform(12.0, 30.0)
        envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
        trace += rng.uniform(0.6, 1.2) * envelope * np.sin(
            2 * np.pi * frequency * (t - center)
        )
    return trace


def _vehicle_trace(rng: np.random.Generator, size: int, sr: int) -> np.ndarray:
    t = np.arange(size) / sr
    trace = 0.45 * _colored_noise(rng, size, 0.25)
    frequency = rng.uniform(28.0, 48.0)
    modulation = 1.0 + 0.25 * np.sin(2 * np.pi * rng.uniform(1.0, 3.0) * t)
    trace += 0.8 * modulation * signal.chirp(
        t, f0=frequency, f1=frequency + rng.uniform(5.0, 15.0),
        t1=max(t[-1], 1.0), method="linear"
    )
    return trace


def _noise_trace(rng: np.random.Generator, size: int, sr: int) -> np.ndarray:
    del sr
    low = _colored_noise(rng, size, 0.12)
    return 0.9 * low + 0.18 * rng.normal(size=size)


def generate_synthetic_traces(
    *,
    traces_per_class: int = 30,
    samples_per_trace: int = 10_000,
    sampling_rate: int = 1_000,
    seed: int = 42,
    label_noise: float = 0.0,
) -> list[dict]:
    """Generate independent traces for noise, human, and vehicle classes.

    Each trace receives a unique subject ID so group-based splitting remains
    leakage-safe when synthetic data is combined with real recordings.
    """
    if traces_per_class < 1 or samples_per_trace < 1:
        raise ValueError("traces_per_class and samples_per_trace must be positive")
    if not 0.0 <= label_noise < 1.0:
        raise ValueError("label_noise must be in the range [0, 1)")

    rng = np.random.default_rng(seed)
    builders = {
        "noise": _noise_trace,
        "human": _human_trace,
        "vehicle": _vehicle_trace,
    }
    traces = []
    for label, builder in builders.items():
        for index in range(traces_per_class):
            traces.append({
                "data": builder(rng, samples_per_trace, sampling_rate),
                "subject": f"synthetic_{label}_{index}",
                "label": label,
                "sr": sampling_rate,
                "recording_id": f"synthetic_{label}_{index}",
            })
    if label_noise:
        labels = list(builders)
        for trace in traces:
            if rng.random() < label_noise:
                alternatives = [item for item in labels if item != trace["label"]]
                trace["label"] = alternatives[int(rng.integers(len(alternatives)))]
    return traces
