import tensorflow as tf
from .stft import build_stft_layer

def build_lstm_model(config):
    input_layer = tf.keras.layers.Input(shape=(1000, 1))
    
    stft = build_stft_layer(config['dataset']['stft_frame_length'], config['dataset']['stft_frame_step'])(input_layer)
    
    lstm_units = config['model']['lstm_units']
    lstm = tf.keras.layers.LSTM(lstm_units)(stft)
    
    dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm)
    drop1 = tf.keras.layers.Dropout(0.5)(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    output = tf.keras.layers.Dense(3, activation='softmax')(drop2)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    
    return model
