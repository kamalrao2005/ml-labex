import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
# Load binary classification dataset (Setosa vs Versicolor)
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Gradient Descent based Logistic Regression
def sigmoid(z): return 1 / (1 + np.exp(-z))
w = np.zeros(X_train.shape[1])
b = 0
lr = 0.01
epochs = 2000
for _ in range(epochs):
    z = X_train @ w + b
    y_hat = sigmoid(z)
    error = y_hat - y_train
    w -= lr * X_train.T @ error / len(y_train)
    b -= lr * np.mean(error)
  # Predict using GD
y_prob_gd = sigmoid(X_test @ w + b)
y_pred_gd = (y_prob_gd >= 0.5).astype(int)
acc_gd = accuracy_score(y_test, y_pred_gd)
conf_gd = confusion_matrix(y_test, y_pred_gd)
fpr_gd, tpr_gd, _ = roc_curve(y_test, y_prob_gd)
auc_gd = roc_auc_score(y_test, y_prob_gd)

# Sklearn Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_lib = model.predict(X_test)
y_prob_lib = model.predict_proba(X_test)[:, 1]

acc_lib = accuracy_score(y_test, y_pred_lib)
conf_lib = confusion_matrix(y_test, y_pred_lib)
fpr_lib, tpr_lib, _ = roc_curve(y_test, y_prob_lib)
auc_lib = roc_auc_score(y_test, y_prob_lib)

# Results
print("Gradient Descent Accuracy:", acc_gd)
print("Gradient Descent Confusion Matrix:\n", conf_gd)
print("Gradient Descent AUC:", auc_gd)

print("\nLibrary Accuracy:", acc_lib)
print("Library Confusion Matrix:\n", conf_lib)
print("Library AUC:", auc_lib)
# ROC Curve Plot
plt.plot(fpr_gd, tpr_gd, label=f'GD AUC = {auc_gd:.2f}', linestyle='--')
plt.plot(fpr_lib, tpr_lib, label=f'Library AUC = {auc_lib:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Gradient Descent vs Library")
plt.legend()
plt.grid()
plt.show()
