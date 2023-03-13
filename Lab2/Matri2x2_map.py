import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = {'y_Actual': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No'],
        'y_Predicted': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No']
        }
df = pd.DataFrame(data, columns= ['y_Actual', 'y_Predicted'])
df['a_Actual'] = df['y_Actual'].map({'Yes': 1, 'No': 0 })
df['a_Predicted'] = df['y_Predicted'].map({'Yes': 1, 'No': 0 })

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'],
                               rownames=['Actual'], colnames= ['Predicted'])

#The plot is the same as the one from Matrix2x2
sn.heatmap(confusion_matrix, annot=True)
plt.show()
