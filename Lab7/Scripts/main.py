import numpy as np
import matplotlib.pyplot as plt


def J_cost(y_efectiv, y_estimat):
    #Functia cost -eroarea medie patrata
    cost = np.sum((y_efectiv-y_estimat)**2)/len(y_efectiv)
    return cost


def GD(x, y, iteratii= 1000, rata_invatare = 0.0001, stop_prag=1e-6):
    w_curent = 0.1 #ponderea start
    b_curent = 0.01 #bias start
    iteratii = iteratii
    rata_invatare = rata_invatare
    n = float(len(x))
    pierderi = []
    ponderi_w = []
    pierdere_anterioara = None

    cost_anterior = 0
    #Estimare parametri optimi
    for i in range(iteratii):
        #Estimati
        y_estimat = (w_curent*x) + b_curent
        #Calcul functie cost curenta
        cost_curent = J_cost(y, y_estimat)
        if cost_anterior and abs(cost_anterior-cost_curent)<=stop_prag:
            break

        cost_anterior = cost_curent

        pierderi.append(cost_curent)
        ponderi_w.append(w_curent)

        #Calcul gradienti
        w_derivata = -(2/n)* sum(x*(y-y_estimat))
        b_derivata = -(2/n)* sum(y-y_estimat)

        #Actualizare w si b
        w_curent = w_curent - (rata_invatare * w_derivata)
        b_curent = b_curent - (rata_invatare * b_derivata)

        #Tiparim rezultate
        print(f"Iteratia {i+1}:")#f = formated string literals/ {}
        print(f"Cost {cost_curent}\nPondere{w_curent}\nBias {b_curent}","\n")

    #Vizualizare grafica
    plt.figure(figsize = (8.6))
    plt.plot(ponderi_w, pierderi, color="green", linewidth=2, linestyle=':')
    plt.scatter(ponderi_w, pierderi, marker='o', color= 'magenta')
    plt.title("Cost vs Pomnderi", color= "blue")
    plt.ylabel("Cost", color= "red")
    plt.xlabel("Ponderi", color="red")
    plt.legend(["Linia intre puncte", "Perechi J-w"], loc= "upper right")
    plt.show()

    return w_curent, b_curent


def m():
    #Datele de luctu
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787, 55.14218841, 52.21179669, 39.29956669,
                  48.10504169, 52.55001444, 45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806, 48.95588857,
                  44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513, 78.21151827, 79.64197305, 59.17148932,
                  75.3312423, 71.30087989, 55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216, 60.72360244,
                  82.89250373, 97.37989686, 48.84715332, 56.87721319])

    w_estimat, b_estimat = GD(X, Y, iteratii=2000)
    print(f"Pondere estimata (w): {w_estimat}\nBias estimat (b): {b_estimat}")

    #Predictii
    Y_pred = w_estimat*X + b_estimat

    #Grafic
    plt.figure(figsize= (8,6))
    plt.scatter(X, Y, marker='o', color='red', edgecolors="blue")
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='navy', markerfacecolor='red', markersize=10, linestyle='dashed')
    plt.title("Modelul de regresie liniara creat cu descendenta in gradient", color= "magenta")
    plt.xlabel("Date pe X", color="blue")
    plt.ylabel("Date pe Y", color="blue")
    plt.legend(["Modelul", "Perechi X-Y"], loc="lower right")
    plt.show()


m()
