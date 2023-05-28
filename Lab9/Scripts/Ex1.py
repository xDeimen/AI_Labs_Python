import numpy as np
import pandas as pd

cale = "C:\\Users\\guias\\Desktop\\Laboratoare AI\\Laboratory 9 -"
file = "HR_comma_sep.csv"
data = pd.read_csv(cale+"\\"+file)

data.head()

from sklearn import preprocessing
# alocam etichete numerice
le = preprocessing.LabelEncoder()
# inlocuim valori string cu etichete
data['salary']=le.fit_transform(data['salary'])
data['sales']=le.fit_transform(data['sales'])
data.head()

X = data[['satisfaction_level','last_evaluation','number_project','average_monthly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']]
y = data['left']

from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.3,random_state=42)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(6,5),random_state=5,verbose=True,learning_rate_init=0.01)

clf.fit(X_train,X_test)

ypred = clf.predict(X_test)
print(ypred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,ypred)

import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
plt.scatter(y_test,ypred,color="red",marker="o")
plt.show()
print(len(y_test),type(y_test))
print(len(ypred),type(ypred))

c=0
for i in range(0,len(y_test)):
    if y_test[i]!=ypred[i]:
        c+=1
print("Numar de predictii eronate= ", c," din ",len(ypred)," predictii")