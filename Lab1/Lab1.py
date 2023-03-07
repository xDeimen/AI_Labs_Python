import pandas as pd
import numpy as np

input_dim = 3 #numarul de intrari
learning_rate = 0.01
#start ponderi in mod aleator (3 valori)

Weights = np.random.rand(input_dim)

Weights[0]=0.5
Weights[1]=0.5
Weights[2]=0.5


#extragem datele de instruire

Training_Data = pd.read_csv("S:\\III\\Sem II\\AI\\AI_Labs_Python\\Lab1\\Data_Set.csv")
print(Training_Data)
Expected_Output = Training_Data.output
Training_Data = Training_Data.drop(['output'], axis=1)
Training_Data = np.asarray(Training_Data)

training_count = len (Training_Data[:,0])

for epoch in range(0,100):
    for datum in range(0, training_count):
        Output_Sum = np.sum(np.multiply(Training_Data[datum,:], Weights))
        #functia de activare de tip treapta

        if Output_Sum < 0:
            Output_Value = 0
        else:
            Output_Value = 1

        error = Expected_Output[datum] - Output_Value
        for n in range(0, input_dim):
            Weights[n]= Weights[n] + learning_rate*error*Training_Data[datum,n]

print("w_0=%.3f"%(Weights[0]))
print("w_0=%.3f"%(Weights[1]))
print("w_0=%.3f"%(Weights[2]))

def aplicare(d_intrare):
    element = 0
    for i in range(0,input_dim):
        element = element + d_intrare[i]*Weights[i]
    #functia de activare
    if element < 0:
        iesire = 0
    else:
        iesire = 1
    print("Pentru intrarea", d_intrare, ", iesirea este", iesire)

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i+1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

#Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range (n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row,weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' %(epoch, l_rate, sum_error))
    return weights

# test predictions
dataset = [[12.7810836,2.550537003,0],
        [11.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235, 1],
        [15.332441248,2.088626775, 1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655, 1],
        [17.673756466,3.508563011,1]]

weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
for row in dataset:
    prediction = predict(row, weights)
print("Expected %d, Predicted=%d" % (row[-1], prediction))

l_rate = 0.1
n_epoch = 5
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)