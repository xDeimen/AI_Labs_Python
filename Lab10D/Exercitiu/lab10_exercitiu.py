# -*- coding: utf-8 -*-
"""Lab10D Exercitiu.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SPZHq_AIB8KvDYYTL4F7fdcxwT_MdNJe
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

columns = ['sarcini', 'glucoza', 'tensiune', 'pliu_cutanat', 'insulina', 'BMI', 'pedigree', 'varsta', 'class']
df = pd.read_csv('/content/drive/Othercomputers/Omen/III/Sem II/AI/LUCRARI_10/clasificare_diabet_gen_F.csv', names=columns)

print(df.head())

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
print("\ntrain\n----------------\n",train.head())
print("\nvalid\n----------------\n",valid.head())
print("\ntest\n----------------\n",test.head() )

#Verificam nevoia de oversample/undersample
#pt ca sunt inegale numerele de clase o sa fie un bias catre cele cu 0
#asa ca o sa dam oversample la cele din clasa 1
#also le normalizam
ones = 0
zeros = 0

for i in train["class"]:
  if i == 1:
    ones = ones+1
  else:
    zeros = zeros+1

print("Ones: ", ones)
print("Zeros: ", zeros)

#Clasa pentru normalizare si oversample daca este cazul
def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X,y)
  

  data = np.hstack((X, np.reshape(y,(-1,1))))

  return data, X, y

#impartim train valid test in x si y si le scalam si pt train folosim si oversampler
train , X_train, y_train = scale_dataset(train, oversample=True)
valid , X_valid, y_valid = scale_dataset(valid, oversample=False)
test , X_test, y_test = scale_dataset(test, oversample=False)

#print to see if scaler works
print("\ntrain\n----------------\n",train)

#Keras model from the assigmnet

def train_model(X_train, y_train, lr, batch_size, epochs):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, input_dim=8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

  return model, history

#Plotting functions
def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)
  plt.show()

def plot_accuracy(history):
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label='val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()

#Training the model
#params data, output, learning rate, batchsize, epochs
model, history = train_model(X_train, y_train, 0.001, 32, 100)

plot_history(history)
plot_accuracy(history)

#predictii pe test sample
from sklearn.metrics import classification_report

predictions = model.predict(X_test)
plt.clf()
plt.plot(predictions, color="green")
plt.xlabel("Id of dataframe")
plt.ylabel("Last neuron fired with")

#We now make a report based on the test set 
#the metrics are weak, so for better metrics, tuning the parameters is in order
predictions = (predictions > 0.5).astype(int).reshape(-1,)
print(classification_report(y_test, predictions))
