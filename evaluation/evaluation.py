
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

a="diamon"   

y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])  
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0])
y_prob = np.array([0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.45])

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

print("\nClassification Report:\n", classification_report(y_true, y_pred))



y_true_reg = np.array([3.0, 5.0, 7.0, 10.0])
y_pred_reg = np.array([2.8, 5.1, 6.9, 9.8])

print("Mean Squared Error (MSE) :", mean_squared_error(y_true_reg, y_pred_reg))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_true_reg, y_pred_reg))
print("R² Score                :", r2_score(y_true_reg, y_pred_reg))





