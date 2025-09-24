# neural network with framework 
#                                                tensorflow
# it easy



# import numpy as np
# from tensorflow.keras.models import Sequential
# from  tensorflow.keras.layers import Dense

# X=np.array([[1,0],[0,1]])
# y=np.array([[1],[0]])

# model=Sequential([
#     Dense(5,input_dim=2,activation="relu"),
#     Dense(1,activation="sigmoid")
# ])

# model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# model.fit(X,y,epochs=500,verbose=0)

# pred=model.predict(X)
# for i,p in enumerate(pred):
#     print(f"input{X[i]}--> prediction -->{p[0]:.3f}")




#                                    pytorch

import torch 
import torch.nn as nn
import torch.optim as optim

X=torch.tensor([[1,0],[0,1]],dtype=torch.float32)
y=torch.tensor([[1],[0]],dtype=torch.float32)

