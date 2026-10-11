import sys, inspect
from tensorflow.keras import layers, Model
import tensorflow as tf
from keras_lmu import LMU

# 1. Inspect LMU signature and source
print('--- LMU signature ---')
print(inspect.signature(LMU))
print('\n--- LMU __init__ source (truncated) ---')
source = inspect.getsource(LMU.__init__)
print(source)

# 2. Build model according to Hazan spec
input_tensor = layers.Input(shape=(1000, 1), name='signal')
# STFT
stft = layers.Lambda(lambda x: tf.signal.stft(tf.squeeze(x, -1), frame_length=256, frame_step=128, fft_length=256), name='stft')(input_tensor)
mag = layers.Lambda(lambda s: tf.abs(s), name='mag')(stft)
# Determine correct LMU constructor args from signature
# Assuming signature includes: memory_dim, order, hidden_dim (or units) etc.
# The Hazan spec: memory dimension = 8, order = 212, hidden state = 16
# We'll map these to the appropriate arguments by name.
# For robustness, we inspect parameter names.
param_names = [p.name for p in inspect.signature(LMU).parameters.values()]
print('\n--- LMU parameter names ---')
print(param_names)

# Build LMU layer using keyword arguments matched to spec
lmu_kwargs = {}
# Map known spec to possible param names
for name in param_names:
    if name.lower() in ['memory_dim', 'memory', 'm', 'memsize']:
        lmu_kwargs[name] = 8
    elif name.lower() in ['order', 'order_', 'n', 'order_dim']:
        lmu_kwargs[name] = 212
    elif name.lower() in ['hidden_dim', 'hidden', 'units', 'output_dim', 'output']:
        # Use 16 for hidden/output dimension
        lmu_kwargs[name] = 16
    elif name.lower() in ['activation']:
        lmu_kwargs[name] = 'tanh'
# Create LMU layer
lmu_layer = LMU(**lmu_kwargs, name='lmu')
# Apply LMU (expects 2D input: time, features) – we flatten across time steps
# LMU expects shape (batch, time, features); our mag is (batch, time, freq) where freq=129
lmu_out = lmu_layer(mag)

# Classification head
x = layers.Dense(128, activation='relu', name='dense128')(lmu_out)
x = layers.Dropout(0.5, name='dropout1')(x)
x = layers.Dense(64, activation='relu', name='dense64')(x)
x = layers.Dropout(0.3, name='dropout2')(x)
output = layers.Dense(3, activation='softmax', name='output')(x)

model = Model(inputs=input_tensor, outputs=output, name='LMU_Hazan')

print('\n--- Model Summary ---')
model.summary()
# Print per-layer parameter counts
print('\n--- Per-layer parameter counts ---')
for layer in model.layers:
    print(f"{layer.name}: {layer.count_params()} params")

# Total params
print('\nTotal params (model.count_params()):', model.count_params())
