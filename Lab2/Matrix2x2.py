import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = {'y_Actual': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }
df = pd.DataFrame(data, columns= ['y_Actual', 'y_Predicted'])
print(df)
print()

# To get different .png with All column and row, we add margins = True, to crosstab function
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'],
                               rownames=['Actual'], colnames= ['Predicted'], margins= True)

print(confusion_matrix)

sn.heatmap(confusion_matrix, annot=True)
plt.show()
