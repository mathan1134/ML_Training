# # bagging
# # reduce varience like overfitting and stablity

# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
# from sklearn.datasets import load_iris

# iris=load_iris()
# X=iris.data[iris.target !=2]
# y=iris.target[iris.target !=2]

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# bagging=BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=50,random_state=42)

# bagging.fit(X_train,y_train)
# baggi=bagging.predict(X_test)

# print("acuracy of bagging :",accuracy_score(y_test,baggi))


# # boosting
# # when model is too simple, to reduce bias & improve accuracy

# boosting=AdaBoostClassifier(
#     estimator=DecisionTreeClassifier(max_depth=1),
#     n_estimators=50,
#     learning_rate=1.0,
#     random_state=42
# )

# boosting.fit(X_train,y_train)
# boost=boosting.predict(X_test)

# print("acuracy of boosting :",accuracy_score(y_test,boost))





#XGboost 
# not use tiny data
# also clear nun values and accuracy improvement 



import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y=load_breast_cancer(return_X_y=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
xb=xgb.XGBClassifier(
    n_estimators=200,        # number of boosting rounds
    learning_rate=0.1,       # step size shrinkage
    max_depth=4,             # tree depth
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'

)
xb.fit(X_train,y_train)
yp_pred=xb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, yp_pred))



# lightGbm
# dataset is very large or has many categorical features.

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=-1,         # -1 means no limit
    num_leaves=31,        # complexity control
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))