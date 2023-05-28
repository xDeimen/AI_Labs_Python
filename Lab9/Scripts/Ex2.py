import numpy as np
import matplotlib.pyplot as plt
# install neurolab
import neurolab as nl

# Generam date
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val,max_val,num_points)
y = 3*np.square(x)+5
y /= np.linalg.norm(y)

# Pregatim datele
intrari = x.reshape(num_points,1)
iesiri = y.reshape(num_points,1)

# Vizualizam rezultatele
plt.figure(figsize=(8,5))
plt.plot(intrari,iesiri,color="red",marker="o")
plt.xlabel('Dimensiunea 1')
plt.ylabel('Dimensiunea 2')
plt.title('Date intrare')

# Cream un MLP

nn = nl.net.newff([[min_val,max_val]],[10,6,1])

# Apelam la GD pentr instruire
nn.trainf = nl.train.train_gd

# Instruim si vizualizam din 100 in 100
error_progress = nn.train(intrari,iesiri, epochs = 2000, show = 100, goal = 0.01)

# Aplicam modelul pe date
output = nn.sim(intrari)
y_pred = output.reshape(num_points)

# Afisam eroarea
plt.figure(figsize=(8,5))
plt.plot(error_progress,color="green")
plt.xlabel('Numar de epoci')
plt.ylabel('Eroarea')
plt.title('Progres eroare')

# Afisam rezultatul
x_dense = np.linspace(min_val,max_val,num_points*2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)
plt.figure(figsize=(8,5))
plt.plot(x_dense,y_dense_pred,'-',x,y,'.',x,y_pred,'b')
plt.title('Actual vs Estimat')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(x_dense,y_dense_pred,'-r',label="estimat")
plt.plot(x,y,'.b',label="actual")
plt.legend()
plt.title('Actual vs Estimat')
plt.show()