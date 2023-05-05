import numpy as np

"""
Exercitiu | descifrati matricea de mai jos
Calculati acuratetea, precizia si sensitivitatea
pentru clasele 0 si 4.
"""

cm = np.array(
[[0.801, 0.104,  0.009, 0.006, 0.031, 0.041, 0.008],
 [0.104, 0.780,  0.007, 0.003, 0.057, 0.012, 0.037],
 [0.015, 0.020,  0.898, 0.012, 0.016, 0.030, 0.009],
 [0.032, 0.003,  0.016, 0.910, 0.002, 0.029, 0.008],
 [0.080, 0.108,  0.012, 0.013, 0.738, 0.022, 0.027],
 [0.078, 0.021,  0.061, 0.021, 0.030, 0.760, 0.029],
 [0.037, 0.106,  0.016, 0.006, 0.051, 0.038, 0.746]])

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label]/col.sum()

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions/rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range (columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls/columns

print("label precision recall")
for label in range(7):
    print(f"{label:5d}{precision(label,cm):9.3f}{recall(label, cm):6.3f}")

print("precision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

print("accuracy: ", accuracy(cm))