import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_lmu import LMU

# Build model as described
input_tensor = layers.Input(shape=(1000,1), name='signal')
# STFT (6,129)
stft = layers.Lambda(lambda x: tf.signal.stft(tf.squeeze(x,-1), frame_length=256, frame_step=128, fft_length=256), name='stft')(input_tensor)
mag = layers.Lambda(lambda s: tf.abs(s), name='mag')(stft)

# Attempt LMU with memory_d=8, order=212, hidden_cell=layers.Dense(16)
try:
    lmu_layer = LMU(memory_d=8, order=212, theta=1.0, hidden_cell=layers.Dense(16), name='lmu')
    out = lmu_layer(mag)
    model = Model(inputs=input_tensor, outputs=out)
    model.summary()
    print('Construction succeeded')
    print('LMU output shape:', out.shape)
    for layer in model.layers:
        if layer.name.startswith('lmu'):
            print('LMU params:', layer.count_params())
except Exception as e:
    print('Construction failed:', e)
