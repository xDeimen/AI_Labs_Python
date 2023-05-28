# -*- coding: utf-8 -*-
"""AI Lab10D Ex1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sIYD2RxnpDmXMSQX_vkUVBeGjbvTr4vZ
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Sneaker', 'Bag', 'Ankle boot']

print("Setul de instruire", train_images.shape)
print("Numarul de imagini in setul de instruire", len(train_labels))
print("Cum arata etichetele din setul de instruire", train_labels)
print("Setul de testare", test_images.shape)
print("Numarul de imagini in setul de date", len(test_labels))

plt.figure(1)
plt.imshow(train_images[0])
plt.figure(2)
plt.imshow(train_images[1])
plt.figure(1002)
plt.imshow(train_images[1002])
plt.colorbar()
plt.grid(False)
plt.show()

"""
Scalam valorile = normalizare
"""

train_images = train_images / 255.0
test_images = test_images/ 255.0

#afisam primele 30 imagini din clasa de instruire
plt.figure("Din set instruire", figsize=(10,10))
for i in range(30):
  plt.subplot(5,6,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()

#afisam 25 de imagini din setul de testare de la 1000 la 1025
plt.figure("Din set testare", figsize=(10,10))
i = 1000
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(test_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[test_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer= 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

#evaluare acuratete
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest accuracy", test_acc)

#de aici modelul este anrentat si se poate folosi pentru predictii
"""iesirea este de tip logits de aceea se face o transformare
inprobabilitati, care sunt mai usor de interpretat"""
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

#pentru prima imbracaminte din lista
print(predictions[0])

print(np.argmax(predictions[0]))

#pentru imbracamintea 100 din lista

print(predictions[99])
print(np.argmax(predictions[99]))

#examinarm eticheta de testare
print(test_labels[0])
print(test_labels[99])

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{}{:2.0f}% ({})".format(class_names[predicted_label],
                                      100*np.max(predictions_array),
                                      class_names[true_label]),
                                      color=color)
  
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

"""Modelul odata instruit poate fii folosit pentru a face predictii despre unele imagini.
Sa analizam prima sia 13-a imagine, predictiile si matricea de predictii
Etichetele de predictie corecte sunt albastre si etichetele de predictie incorecte sunt rosii
Numarul indica procentajul (din 100) pentru eticheta prezisa."""

i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i=12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#afisam mai multe imagini
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

img = test_images[1]
print(img.shape)

img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_=plt.xticks(range(11), class_names, rotation=45)
print(np.argmax(predictions_single[0]))
