import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#intializam RandomState
#seed(2) doua array-uri

np.random.seed(2)
#distributie normala de numere aletoare
'''(3,1,100) - media aritmetica 3, deviatia standard 1 si numarul de date in array 100'''

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

print(x)
print(y)
'''slicing array x[start:stop:step]'''
#selectie seturi de instruire
train_x = x[:80]
train_y = y[:80]

#selectie seturi de testare
test_x = x[80:]
test_y = y[80:]

#afisare grafica
plt.subplot(231)#2 linii, 3 coloane, prima poz
plt.title("Set creat")
plt.scatter(x, y, color="deeppink")
plt.subplot(232)
plt.title("Set selectat pentru instruire")
plt.scatter(train_x, train_y, color="blue")
plt.subplot(233)
plt.title("Set selectat pentru testare")
plt.scatter(test_x, test_y, color="lightgreen")
plt.subplots()
plt.title("toate trei la un loc: originalul nu se vede ca urmare a suprapunerii")
plt.scatter(x, y, color="deeppink")
plt.scatter(train_x, train_y, color="blue")
plt.scatter(test_x, test_y, color="lightgreen")

'''modelul polinomial este valabil: alegem sa fie de gradul 4'''
model = np.poly1d(np.polyfit(train_x, train_y, 4))
orizontala = np.linspace(0, 6, 100)
plt.subplots()
plt.scatter(train_x, train_y, color="blue")
plt.plot(orizontala, model(orizontala), color="red")
plt.show()

'''testam acuratetea modelului'''
r = r2_score(train_y, model(train_x))
print("relevanta modelului=",r)

'''verificam pe setul de testare'''
r_test = r2_score(test_y, model(test_x))
print("relevanta la testare=", r_test)

'''putem utiliza acum modelul pentru predictii'''
intrare = 5
pred_1 = model(intrare)
print("Pentru intrarea", intrare, "predictia la iesire este", pred_1)


