import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_lmu import LMU

# Build the full model as before (STFT -> LMU -> dense head)
inp = layers.Input(shape=(1000,1), name='signal')
stft = layers.Lambda(lambda x: tf.signal.stft(tf.squeeze(x,-1),frame_length=256,frame_step=128,fft_length=256), name='stft')(inp)
mag = layers.Lambda(lambda s: tf.abs(s), name='mag')(stft)
# LMU as used in experiments
lmu = LMU(memory_d=8, order=212, theta=1.0, hidden_cell=layers.Dense(16), name='lmu')
lmu_out = lmu(mag)
# Classification head
x = layers.Dense(128, activation='relu', name='dense128')(lmu_out)
x = layers.Dropout(0.5, name='dropout1')(x)
x = layers.Dense(64, activation='relu', name='dense64')(x)
x = layers.Dropout(0.3, name='dropout2')(x)
out = layers.Dense(3, activation='softmax', name='output')(x)
model = Model(inputs=inp, outputs=out)

# Get LMU layer
lmu_layer = model.get_layer('lmu')
print('LMU layer type:', type(lmu_layer))
print('LMU internal layer:', type(lmu_layer.layer))
# List all variables (weights) of the LMU layer
total = 0
for var in lmu_layer.weights:
    name = var.name
    shape = var.shape
    trainable = var.trainable
    count = tf.size(var).numpy()
    total += count
    print(f"{name} | shape={shape} | trainable={trainable} | params={count}")
print('Sum of LMU params:', total)
