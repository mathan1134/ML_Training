# # classification (supervised)
# # it define acuracy score model well or not and everyone have specific like f1 hue like that

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_breast_cancer
# from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,precision_score,recall_score
# from sklearn.model_selection import train_test_split


# data=load_breast_cancer()
# X,y=data.data,data.target

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# model=LogisticRegression(max_iter=1000)
# model.fit(X_train,y_train)

# pred=model.predict(X_test)

# print("Confusion Matrix:\n", confusion_matrix(y_test, pred))        
# print("Accuracy:", accuracy_score(y_test, pred))                   #handwright     
# print("Precision:", precision_score(y_test, pred))                  #fraud detcection   fp
# print("Recall:", recall_score(y_test, pred))                           #cancer ,disease fn
# print("F1 Score:", f1_score(y_test, pred))                             # nlp    fP ,fn
# print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))          #credit score  overall
 

# regression

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

X,y=make_regression(n_samples=100,n_features=1,noise=15,random_state=42)

model=LinearRegression()
model.fit(X,y)
pred=model.predict(X)

print("MAE:", mean_absolute_error(y, pred))             #house price
print("MSE:", mean_squared_error(y, pred))                #stock price
print("RMSE:", np.sqrt(mean_squared_error(y, pred)))      #stock price
print("R² Score:", r2_score(y, pred))                     #sales trend

