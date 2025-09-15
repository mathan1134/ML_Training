# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression,LogisticRegression
# import matplotlib.pyplot as plt


                                    #   linear regression
# data={
#     "SquareFeet": [800, 1000, 1200, 1500, 1800],
#     "Price": [200000, 250000, 280000, 350000, 400000]
# }
# df=pd.DataFrame(data)

# X=df[["SquareFeet"]]
# y=df[["Price"]]

# model=LinearRegression()
# model.fit(X,y)

# predicted=model.predict(X)

# plt.scatter(X,y ,label="acual",color="blue")
# plt.plot(X,predicted,color="red",label="predicted")
# plt.xlabel("squarefeet")
# plt.ylabel("price")
# plt.legend()
# plt.show()

# predicted_price=model.predict(pd.DataFrame([[1600]],columns=["SquareFeet"]))
# print(predicted_price[0])


                                                                     #Logestic regression

# data={
#     "HoursStudied": [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1]
# }                                          

# df=pd.DataFrame(data)

# X=df[["HoursStudied"]]
# y=df["Pass"]

# model=LogisticRegression()
# model.fit(X,y)

# predicted=model.predict(X)

# find=model.predict(pd.DataFrame([[4.5]],columns=["HoursStudied"]))
# a=find[0]
# if a>0:
#     print("pass")
# else:
#     print("fail")    

# plt.scatter(X,y,label="pass & fail",color="red")
# plt.plot(X,predicted,color="blue",label="predicted")
# plt.scatter(4.5,find,s=100,marker="o",color="black",label="find")
# plt.xlabel("studyhours")
# plt.ylabel("pass or fail")
# plt.legend()
# plt.show()



                                                            #    classification

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# iris=load_iris()
# X,y=iris.data,iris.target
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# # decession tree

# dt=DecisionTreeClassifier()
# dt.fit(X_train,y_train)
# print("decesion tree :",accuracy_score(y_test,dt.predict(X_test)))

# # knn knearesrneighbour
# knn=KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train,y_train)
# print("knn :",accuracy_score(y_test,knn.predict(X_test)))

# # svc
# svm=SVC(kernel="linear")
# svm.fit(X_train,y_train)
# print("svc :",accuracy_score(y_test,svm.predict(X_test)))

# # random forest
# rf=RandomForestClassifier(n_estimators=100)
# rf.fit(X_train,y_train)
# print("random forest :",accuracy_score(y_test,rf.predict(X_test)))





                                                            # titanic dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

titanic=sns.load_dataset("titanic")

data=titanic[["pclass","sex","age","fare","survived"]].dropna()

data["sex"]=LabelEncoder().fit_transform(data["sex"])

X=data.drop("survived",axis=1)
y=data["survived"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


ll=LogisticRegression()
ll.fit(X_train,y_train)
y_pred=ll.predict(X_test)
print("logestic regression:",accuracy_score(y_test,y_pred))

svc=SVC(kernel="linear")      #without kernal low acurracy
svc.fit(X_train,y_train)
print("svm :",accuracy_score(y_test,svc.predict(X_test)))

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print("knn :",accuracy_score(y_test,knn.predict(X_test)))


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("dt :",accuracy_score(y_test,dt.predict(X_test)))

rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
print("rf :",accuracy_score(y_test,rf.predict(X_test)))


nb=GaussianNB()
nb.fit(X_train,y_train)
print("nb :",accuracy_score(y_test,nb.predict(X_test)))
