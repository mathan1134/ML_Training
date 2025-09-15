                           #gridsearchcv 


# They control how the learning happens.
# They are not learned from the data.



# You’re baking a cake 🎂.
# Ingredients = data.
# Recipe steps (oven temperature, baking time, mixing speed) = hyperparameters.
# Cake taste = model performance. 


# from sklearn.datasets import load_iris
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# X,y=load_iris(return_X_y=True)

# model=RandomForestClassifier()

# p_grid={
#     "n_estimators":[50,100,200],
#     "max_depth": [None, 5, 10]
# }

# grid=GridSearchCV(model,p_grid,cv=3)
# grid.fit(X,y)

# print("best param :",grid.best_params_)
# print("best score :",grid.best_score_)


#                                                RandomizedSearchCV

# Test a random subset of combinations.
# faster and optimal results

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
# import numpy as np

# X,y=load_iris(return_X_y=True)
# model=RandomForestClassifier()

# p_dist={
#     "n_estimators":np.arange(50,500,50),
#     "max_depth":(None,5,10,20)
# }

# randomsearchcv=RandomizedSearchCV(model,p_dist,n_iter=3,cv=3,random_state=42)

# randomsearchcv.fit(X,y)

# print("best params :",randomsearchcv.best_params_)
# print("best score :",randomsearchcv.best_score_)



#                                         Bayesian Optimization

# smart search probabillity
# fast and deep learning
# exra libraries need

# import optuna as ot
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import load_iris

# X,y=load_iris(return_X_y=True)

# def objective(trial):
#     n_estimators=trial.suggest_int("n_estimators",50,300)
#     max_depth=trial.suggest_int("max_depth",2,20)

#     model=RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         random_state=42
#     )
#     score=cross_val_score(model,X,y,cv=3).mean()
#     return score
# study=ot.create_study(direction="maximize")
# study.optimize(objective,n_trials=10)

# print("best params :",study.best_params)
# print("best score :",study.best_value)