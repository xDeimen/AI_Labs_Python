import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn

seaborn.set(style='whitegrid');seaborn.set_context('talk')

# vedem ce contine setul de date
from sklearn.datasets import load_iris
iris_data = load_iris()
print(iris_data['DESCR'])

# afisam setul de date
n_samples,n_features = iris_data.data.shape

plt.subplot(1,2,1)
scatter_plot = plt.scatter(iris_data.data[:,0],iris_data.data[:,1],alpha=0.5,c=iris_data.target)
plt.colorbar(ticks=([0,1,2]))
plt.title('Petal Esantion')
plt.show()

# citim datele din fisier si le afisam grafic
import pandas as pd
from pandas.plotting import scatter_matrix as sm
print(pd.__version__)
dataset = pd.read_csv("iris.csv")
feature_1 = ['sepal.length','sepal.width','petal.length','petal.width']

# scatter_matrix este invechita pt ver. peste 0.2 ale pandas
sm(dataset[feature_1],figsize=(20,20))
plt.show()

dataset.hist(figsize = (20,20),color='red')
plt.show()

dataset.plot(subplots=True,figsize=(10,10),sharex=False,sharey=False)
plt.show()

random.seed(123)

def separate_data():
    A=iris_dataset[0:40]
    tA=iris_dataset[40:50]
    B=iris_dataset[50:90]
    tB=iris_dataset[90:100]
    C=iris_dataset[100:140]
    tC=iris_dataset[140:150]
    train = np.concatenate((A,B,C))
    test = np.concatenate((tA,tB,tC))
    return train,test

train_procent = 80
test_procent = 20
iris_dataset = np.column_stack((iris_data.data,iris_data.target.T))
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)
Filetrain, Filetest = separate_data()

train_X = np.array([i[:4]for i in Filetrain])
test_X = np.array([i[:4]for i in Filetest])
train_y = np.array([i[:4]for i in Filetrain])
test_y = np.array([i[:4]for i in Filetest])

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.subplot(1,2,1)
plt.scatter(train_X[:,0],train_X[:,1],c=train_y,cmap = cm.viridis)
plt.xlabel(iris_data.feature_names[0])
plt.ylabel(iris_data.feature_names[1])

plt.subplot(1,2,2)
plt.scatter(train_X[:,2],train_X[:,3],c=train_y,cmap = cm.viridis)
plt.xlabel(iris_data.feature_names[2])
plt.ylabel(iris_data.feature_names[3])

plt.subplot(1,2,1)
plt.scatter(test_X[:,0],test_X[:,1],c=test_y,cmap = cm.viridis)
plt.xlabel(iris_data.feature_names[0])
plt.ylabel(iris_data.feature_names[1])

plt.subplot(1,2,2)
plt.scatter(test_X[:,2],test_X[:,3],c=test_y,cmap = cm.viridis)
plt.xlabel(iris_data.feature_names[2])
plt.ylabel(iris_data.feature_names[3])

from sklearn.base import BaseEstimator,ClassifierMixin,RegressorMixin
import random

class MultiLayerPerceptron(BaseEstimator,ClassifierMixin):
    def __init__(self,params=None):
        if(params==None):
            self.inputLayer=4
            self.hiddenLayer=5
            self.outputLayer=3
            self.learningRate=0.005
            self.max_epochs=600
            self.BiasHiddenValue=-1
            self.BiasOutputValue=-1
            self.activation=self.ativacao['sigmoid']
            self.deriv=self.derivada['sigmoid']
        else:
            self.inputLayer=params['InputLayer']
            self.hiddenLayer=params['HiddenLayer']
            self.outputLayer=params['OutputLayer']
            self.learningRate=params['LearningRate']
            self.max_epochs=params['Epocas']
            self.BiasHiddenValue=params['BiasHiddenValue']
            self.BiasOutputValue=params['BiasOutputValue']
            self.activation=self.ativacao[params['ActivationFunction']]
            self.deriv=self.derivada[params['ActivationFunction']]

        self.WEIGHT_hidden = self.starting_weights(self.hiddenLayer,self.inputLayer)
        self.WEIGHT_output = self.starting_weights(self.outputLayer,self.hiddenLayer)
        self.BIAS_hidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.outputLayer)])
        self.classes_number = 3
    pass

    def starting_weights(self,x,y):
        return[[2*random.random()-1 for i in range(x)] for j in range(y)]

    ativacao = {'sigmoid':(lambda x:1/(1+np.exp(-x))), 'tanh':(lambda x: np.tanh(x)), 'Relu':(lambda x: x*(x>0))}
    derivada = {'sigmoid':(lambda x:x*(1-x)), 'tanh':(lambda x: 1-x**2), 'Relu':(lambda x: 1*(x>0))}

    def Backpropagation_Algorithm(self,x):
        DELTA_output=[]
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1)*(ERROR_output)*self.deriv(self.OUTPUT_L2))

        arrayStore = []
        for i in range(self.hiddenLayer):
            for j in range(self.outputLayer):
                self.WEIGHT_output[i][j] -=(self.learningRate*(DELTA_output[j]*self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate*DELTA_output[j])

        delta_hidden = np.matmul(self.WEIGHT_output,DELTA_output)*self.deriv(self.OUTPUT_L1)

        for i in range(self.outputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate*(delta_hidden[j]*x[i]))
                self.BIAS_hidden[j] -= (self.learningRate*delta_hidden[j])

    def show_err_graphic(self,v_erro,v_epoca):
        plt.figure(figsize=(9,4))
        plt.plot(v_epoca, v_erro, "m-", color="b", marker=11)
        plt.xlabel("Numar de epoci")
        plt.ylabel("Eroare medie patratica (MSE)")
        plt.title("Minimizare eroare")
        plt.show()

    def predict(self,X,y):
        my_predictions = []
        forward = np.matmul(X,self.WEIGHT_hidden)+self.BIAS_hidden
        forward = np.matmul(forward,self.WEIGHT_output)+self.BIAS_output

        for i in forward:
            my_predictions.append(max(enumerate(i),key=lambda x: x[1])[0])
        print("Numar esantioane | Clasa| Output| Output Dorit")

        for i in range(len(my_predictions)):
            if(my_predictions[i]==0):
                print("id:{} |Iris-Stosa|Output: {}".format(i,my_predictions[i],y[i]))
            elif(my_predictions[i]==1):
                print("id:{} |Iris-Versiclour|Output: {}".format(i,my_predictions[i],y[i]))
            elif(my_predictions[i]==2):
                print("id:{} |Iris-Virginica|Output: {}".format(i,my_predictions[i],y[i]))
        return my_predictions

    def fit(self,X,y):
        count_epoch = 1
        total_error = 0
        n = len(X);
        epoch_array = []
        error_array = []
        W0 = []
        W1 = []

        while(count_epoch <= self.max_epochs):
            for idx,inputs in enumerate(X):
                self.output = np.zeros(self.classes_number)
                self.OUTPUT_L1 = self.activation((np.dot(inputs,self.WEIGHT_hidden)+self.BIAS_hidden.T))
                self.OUTPUT_L2 = self.activation((np.dot(self.OUTPUT_L1,self.WEIGHT_output)+self.BIAS_output.T))
                if(y[idx]==0):
                    self.output = np.array([1,0,0])
                elif(y[idx]==1):
                    self.output = np.array([0,1,0])
                elif(y[idx]==2):
                    self.output = np.array([0,0,1])

            square_error = 0
            for i in range(self.outputLayer):
                erro = (self.output[i]-self.OUTPUT_L2[i])**2
                square_error = (square_error + (0.05*erro))
                total_error = total_error + square_error
            self.Backpropagation_Algorithm(inputs)
            total_error = (total_error/n)
            if((count_epoch%50==0)or(count_epoch==1)):
                print("Epoca",count_epoch," Eroare Totala: ", total_error)
                error_array.append(total_error)
                epoch_array.append(count_epoch)
            W0.append(self.WEIGHT_hidden)
            W1.append(self.WEIGHT_output)
            count_epoch+=1
        self.show_err_graphic(error_array,epoch_array)
        plt.plot(W0[0])
        plt.title('Ponderi ascunse actualizate in timpul instruirii')
        plt.legend('neuron1','neuron2','neuron3','neuron4','neuron5')
        plt.ylabel('Ponderea')
        plt.show()
        plt.plot(W1[0])
        plt.title('Ponderi iesiri actualizate in timpul instruirii')
        plt.legend('neuron1','neuron2','neuron3')
        plt.ylabel('Ponderea')
        plt.show()

        return self

    def show_test():
        ep1=[0,100,200,300,400,500,600,700,800,900,1000,1500,2000]
        h_5=[0,60,70,83.3,93.6,96.7,86.7,86.7,76.7,73.3,66.7,66.7]
        h_4=[0,40,70,63.3,66.7,70,70,70,70,66.7,66.7,43.3,33.3]
        h_3=[0,46.7,76.7,80,76.7,76.7,76.6,73.3,73.3,73.3,73.3,76.7,76.7]
        plt.figure(figsize=(10,4))
        I1, = plt.plot(ep1,h_3,"m-",color='b',label="node-3",marker=11)
        I2, = plt.plot(ep1,h_4,"m-",color='g',label="node-4",marker=8)
        I3, = plt.plot(ep1,h_5,"m-",color='r',label="node-5",marker=5)
        plt.legend(handles=[I1,I2,I3],loc=1)
        plt.xlabel("Numar de epoci");plt.ylabel("% Atinse")
        plt.title("Numari de straturi ascunse - Performanta")

        ep2 = [0,100,200,300,400,500,600,700]
        tanh = [0.18,0.027,0.025,0.022,0.0068,0.0060,0.0057,0.00561]
        sigm = [0.185,0.0897,0.060,0.0396,0.0343,0.0314,0.0296,0.0281]
        Relu = [0.185,0.05141,0.05130,0.05127,0.05124,0.05123,0.05122,0.05121]
        plt.figure(figsize=(10,4))
        I1, = plt.plot(ep2,tanh,"m-",color='b',label="Hyperbolic Tangent",marker=11)
        I2, = plt.plot(ep2,sigm,"m-",color='g',label="Sigmoide",marker=8)
        I3, = plt.plot(ep2,Relu,"m-",color='r',label="ReLu",marker=5)   
        plt.legend(handles=[I1,I2,I3],loc=1)
        plt.xlabel("Epoci");plt.ylabel("Eroare");plt.title("Functii de activare - Performanta")

        fig,ax = plt.subplots()
        names = ["Hyperbolic Tangent", "Sigmoide", "ReLu"]
        x1 = [2.0,4.0,6.0]
        plt.bar(x1[0],53.4,0.4,color='b')
        plt.bar(x1[1],96.7,0.4,color='g')
        plt.bar(x1[2],33.2,0.4,color='r')
        plt.xticks(x1,names)
        plt.ylabel('% Atinse')
        plt.title('Atinse - Functii de activare')
        plt.show()

    show_test()

dictionary = {'InputLayer':4,'HiddenLayer':5,'OutputLayer':3,'Epocas':700,'LearningRate':0.005,'BiasHiddenValue':-1,'BiasOutputValue':-1,'ActivationFunction':'sigmoid'}
Perceptron = MultiLayerPerceptron(dictionary)
Perceptron.fit(train_X,train_y)

prev = Perceptron.predict(test_X,test_y)
hits = n_set = n_vers = n_virg = 0
score_set = score_vers = score_virg = 0
for j in range(len(test_y)):
    if(test_y[j]==0): n_set +=1
    elif(test_y[j]==1): n_vers +=1
    elif(test_y[j]==2): n_virg +=1
for i in range(len(test_y)):
    if(test_y[i]==prev[i]): hits +=1
    if(test_y[i]==prev[i] and test_y[i]==0): score_set+=1
    elif(test_y[i]==prev[i] and test_y[i]==1): score_vers+=1
    elif(test_y[i]==prev[i] and test_y[i]==2): score_virg+=1

hits = (hits/len(test_y))*100
faults = 100-hits

graph_hits = []
print("Procente:","%.2f"%(hits),"% atinse","si","%.2f"%(faults),"% erori")
print("Total esantioane test", n_samples)
print("Iris-Setosa:",n_set,"esantioane")
print("Iris-Vesicolor",n_vers,"esantioane")
print("Iris-Virginica",n_virg,"esantioane")

graph_hits.append(hits)
graph_hits.append(faults)
labels = 'Atinse','Erori';
sizes = [96.5,3.3]
explode = (0,0.14)

fig1,ax1 = plt.subplots();
ax1.pie(graph_hits,explode=explode,colors=['blue','red'],labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)
ax1.axis('equal')
plt.show()

acc_set = (score_set/n_set)*100
acc_vers = (score_vers/n_vers)*100
acc_virg = (score_virg/n_virg)*100

print("- Acuratetea Iris-Setosa","%.2f"%acc_set,"%")
print("- Acuratetea Iris-Versicolor","%.2f"%acc_vers,"%")
print("- Acuratetea Iris-Virginica","%.2f"%acc_virg,"%")
names = ["Setosa","Versicolour","Virginica"]
x1 = [2.0,4.0,6.0]
fig,ax = plt.subplots()
r1 = plt.bar(x1[0],acc_set,color='orange',label='Iris-Setosa')
r2 = plt.bar(x1[1],acc_vers,color='green',label='Iris-Versicolour')
r3 = plt.bar(x1[2],acc_virg,color='purple',label='Iris-Virginica')
plt.ylabel('Scoruri %')
plt.xticks(x1,names);plt.title('Scoruri pe irisul florilor - MLP')
plt.show()