from sklearn import metrics

y_pred = ["a", "b", "c", "a", "b"]
y_act = ["a", "b", "c", "c", "a"]

print(metrics.confusion_matrix(y_act, y_pred, labels=["a", "b", "c"]))
print(metrics.classification_report(y_act, y_pred, labels=["a", "b", "c"]))