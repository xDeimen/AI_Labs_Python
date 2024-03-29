import numpy as np
from sklearn import preprocessing

#Labelling data
# Sample input labels
input_labels = ['red','black','red','green','black','yellow','white']

# Creating the label encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Encoding a set of labels
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("Labels =", test_labels)
print("Encoded values =", list(encoded_values))

# Decoding a set of values
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\nDecoded labels =", list(decoded_list))