
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import sys
import tensorflow as tf
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

seed = int(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
try:
    tf.keras.utils.set_random_seed(seed)
except AttributeError:
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "data", "final_dataset.csv"))


df['Return'] = df['Close'].pct_change()
df['MA_20_ratio'] = df['Close'] / df['MA_20'] - 1
df['Close_Open'] = df['Close'] / df['Open'] - 1
df['High_Low'] = df['High'] / df['Low'] - 1
df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean() - 1
df['Volatility'] = df['Return'].rolling(10).std()
df.dropna(inplace=True)

features = ['RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open', 'High_Low', 'Volume_ratio', 'Volatility']
X = df[features]
y = df['Target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

seq_len = 5
X_seq, y_seq = [], []
for i in range(len(X_scaled) - seq_len + 1):
    X_seq.append(X_scaled[i:i+seq_len])
    y_seq.append(y.iloc[i + seq_len - 1])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=seed
)

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=80, batch_size=16, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

print(f"RESULT:{train_acc:.4f}:{accuracy:.4f}")
