import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_lmu import LMU
import inspect, sys

# Inspect LMU signature
sig = inspect.signature(LMU)
print('LMU signature:', sig)
print('Parameter names:', [p.name for p in sig.parameters.values()])

# Build model per Hazan spec
input_tensor = layers.Input(shape=(1000, 1), name='signal')
# STFT
stft = layers.Lambda(lambda x: tf.signal.stft(tf.squeeze(x, -1), frame_length=256, frame_step=128, fft_length=256), name='stft')(input_tensor)
mag = layers.Lambda(lambda s: tf.abs(s), name='mag')(stft)
# Determine theta: total duration = 1 sec (1000 samples at 1kHz)
# Use theta = 1.0 (matches 1 second window)
theta = 1.0
# Create LMU layer with correct args
lmu_layer = LMU(memory_d=129, order=212, theta=theta, hidden_cell=layers.Dense(16), name='lmu')
# Apply LMU
lmu_out = lmu_layer(mag)
# Classification head
x = layers.Dense(128, activation='relu', name='dense128')(lmu_out)
x = layers.Dropout(0.5, name='dropout1')(x)
x = layers.Dense(64, activation='relu', name='dense64')(x)
x = layers.Dropout(0.3, name='dropout2')(x)
output = layers.Dense(3, activation='softmax', name='output')(x)

model = Model(inputs=input_tensor, outputs=output, name='LMU_Hazan')

# Print model summary and per-layer params
model.summary()
print('\nPer-layer parameter counts:')
for layer in model.layers:
    print(f"{layer.name}: {layer.count_params()}")
print('\nTotal params:', model.count_params())
