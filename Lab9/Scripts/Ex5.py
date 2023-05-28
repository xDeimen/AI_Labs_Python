
from pandas import read_csv as rc
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
df = rc(url,header=None)
print(df.shape)
print(df.head())

from matplotlib import pyplot as plt
print(df.describe())
df.hist(color="red")
plt.show()

from sklearn.model_selection import train_test_split as tts
X,y = df.values[:,:-1], df.values[:,-1]
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2)

n_features = X.shape[1]
print(n_features)

from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(max_iter=10000).fit(X_train,y_train)
regr.predict(X_test[:2])
y_pred = regr.predict(X_test)
regr.score(X_test,y_test)

score = mean_absolute_error(y_test,y_pred)
print('MAE: %.3f' % score)
plt.figure(figsize=(8,8))
plt.scatter(y_test,y_pred,color="magenta",marker="o")
plt.title('Harta aproximarii')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

lst=list(range(1,len(y_pred)+1))
LY=[]
for i in range(0,len(y_test)):
    x=y_test[i]-y_pred[i]
    LY.append(x)
plt.figure(figsize=(8,8))
plt.plot(lst,y_test,color="blue",marker="o",label="y_test")
plt.plot(lst,y_pred,color="magenta",marker="o",label="y_pred")
plt.plot(lst,LY,color="green",marker="o",label="eroare")
plt.legend()
plt.show()
