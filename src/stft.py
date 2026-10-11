import tensorflow as tf

def build_stft_layer(frame_length=256, frame_step=128):
    # This function wraps STFT inside a TensorFlow Lambda layer 
    # to be directly embedded in the Keras model, as requested.
    def stft_fn(x):
        # x is expected to be (batch, 1000, 1)
        x = tf.squeeze(x, axis=-1)
        stft = tf.signal.stft(x, frame_length=frame_length, frame_step=frame_step, fft_length=256)
        return tf.abs(stft)
    
    return tf.keras.layers.Lambda(stft_fn, name='Lambda_STFT')

def compute_stft_np(waveform, frame_length=256, frame_step=128):
    import scipy.signal as signal
    import numpy as np
    f, t, Zxx = signal.stft(waveform, fs=1000, nperseg=frame_length, noverlap=frame_length-frame_step, nfft=256)
    return np.abs(Zxx)
