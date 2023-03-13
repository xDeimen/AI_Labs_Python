# Matricea de confuzie
import sklearn

from sklearn.metrics import confusion_matrix
#Acuratetea
from sklearn.metrics import accuracy_score
#Rechemarea
from sklearn.metrics import recall_score
#Precizia
#Scorul F1
from sklearn.metrics import f1_score
# Acuratetea
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Precizia
# Scorul F1
from sklearn.metrics import f1_score
# Rechemarea
from sklearn.metrics import recall_score

y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
y_true = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]

confuzia = confusion_matrix(y_true,y_pred)
print("Matricea de confuzie = \n", confuzia)
acuratetea = accuracy_score(y_true,y_pred)
print("Acuratetea = \n", acuratetea)

#Din None schimba pe micro, observam diferentele
rechemarea = recall_score(y_true,y_pred, average=None)
print("Recall-ul = \n", rechemarea)
factorul_1 = f1_score(y_true,y_pred, average=None)
print("F1 = \n", factorul_1)