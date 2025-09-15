# train & test

# huge dataset it easy 
     

# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# X,y=load_iris(return_X_y=True)

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# model=RandomForestClassifier(random_state=42)
# model.fit(X_train,y_train)

# pred=model.predict(X_test)

# print("accuracy score :",accuracy_score(y_test,pred))




# train test val

# for medium dataset and most keras and pytorch

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X,y=load_iris(return_X_y=True)

X_temp,X_test,y_temp,y_test=train_test_split(X,y,test_size=0.15,random_state=42)

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.175,random_state=42)

model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

va_pred=model.predict(X_val)

print("accuracy score :",accuracy_score(y_val,va_pred))

tes_pred=model.predict(X_test)
print("accuracy score :",accuracy_score(y_test,tes_pred))
