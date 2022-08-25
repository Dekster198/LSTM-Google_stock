import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Embedding, LSTM
from sklearn.preprocessing import MinMaxScaler
from google.colab import drive

drive.mount('/content/drive/')

df = pd.read_csv('/content/drive/MyDrive/Datasets/Google Stock/GOOG.csv')
df = df[['open', 'close']]

plt.figure()
plt.plot(df['open'], label='Open')
plt.plot(df['close'], label='Close')
plt.legend()
plt.show()

MMS = MinMaxScaler()
df[df.columns] = MMS.fit_transform(df)

training_size = round(len(df) * 0.8)

train_data = df[:training_size]
test_data = df[training_size:]

def create_sequence(dataset):
  sequences = []
  label = []
  start_idx = 0

  for stop_idx in range(50, len(dataset)):
    sequences.append(dataset.iloc[start_idx:stop_idx])
    label.append(dataset.iloc[stop_idx])

    start_idx += 1

  return(np.array(sequences), np.array(label))

x_train, y_train = create_sequence(train_data)
x_test, y_test = create_sequence(test_data)

model = keras.Sequential([
    LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
    LSTM(50),
    Dense(2)
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.evaluate(x_test, y_test)