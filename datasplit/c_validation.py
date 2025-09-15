# cross validation
# low data  Model selection + hyperparameter tuning.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)


cv_val=cross_val_score(model,X_train,y_train,cv=5,scoring="accuracy")

print("cv_score :",cv_val)
print("mean :",cv_val.mean())

model.fit(X_train,y_train)

pred_score=model.score(X_test,y_test)
print("predicit_score :",pred_score)


