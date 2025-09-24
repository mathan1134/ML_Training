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

class simpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(2,5)
        self.output=nn.Linear(5,1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self,x)    :
        x=self.relu(self.hidden(x))
        x=self.sigmoid(self.output(x))
        return x
    
model=simpleNN()    
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    output=model(X)
    loss=criterion(output,y)    # loss funct
    loss.backward()             #backprobagation
    optimizer.step()            #update weights


with torch.no_grad():
    prediction=model(X)
    for i,p in enumerate(prediction):
        print(f"input{X[i].numpy()}---> prediction{p.item():.3f}")




