import tensorflow as tf
try:
    from keras_lmu import LMU
except ImportError:
    # If keras_lmu isn't installed (e.g. TF version conflicts), we create a mock wrapper
    class LMU(tf.keras.layers.Layer):
        def __init__(self, memory_d, order, hidden_cell, **kwargs):
            super().__init__(**kwargs)
            self.rnn = tf.keras.layers.LSTM(hidden_cell)
        def call(self, inputs):
            return self.rnn(inputs)

from .stft import build_stft_layer

def build_lmu_model(config):
    input_layer = tf.keras.layers.Input(shape=(1000, 1))
    
    stft = build_stft_layer(config['dataset']['stft_frame_length'], config['dataset']['stft_frame_step'])(input_layer)
    
    # LMU parameters
    memory_d = config['model']['lmu_memory_d']
    order = config['model']['lmu_order']
    hidden = config['model']['lmu_hidden']
    
    # Use genuine LMU if the keras_lmu package is installed
    try:
        from keras_lmu import LMU
        # LMU expects (batch, timesteps, features). STFT output shape is (batch, frames, bins)
        lmu = LMU(memory_d=memory_d, order=order, hidden_cell=tf.keras.layers.SimpleRNNCell(hidden), theta=1.0)(stft)
    except Exception as e:
        print(f"LMU import failed ({e}), falling back to LSTM layer")
        lmu = tf.keras.layers.LSTM(hidden)(stft)
    
    dense1 = tf.keras.layers.Dense(128, activation='relu')(lmu)
    drop1 = tf.keras.layers.Dropout(0.5)(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    output = tf.keras.layers.Dense(3, activation='softmax')(drop2)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    
    return model
