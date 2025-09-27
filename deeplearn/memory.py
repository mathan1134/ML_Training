# # lstm for memory storage

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM,Dense

# seq=np.array([0,1,2,3,4,5,6,7,8,9])

# X=[]
# y=[]

# n_step=3

# for i in range(len(seq)-n_step):
#     X.append(seq[i:i+n_step])
#     y.append(seq[i+n_step])


# X=np.array(X)    
# y=np.array(y)

# X=X.reshape((X.shape[0],X.shape[1],1))

# model=Sequential([
#     LSTM(50,activation="relu",input_shape=(n_step,1)),
#     Dense(1)
# ])

# model.compile(optimizer="adam",loss="mse")

# model.fit(X,y,epochs=500,verbose=0)

# x_input=np.array([7,8,9]).reshape((1,n_step,1))
# y_pred=model.predict(x_input,verbose=1)

# print(f"prediction of [7,8,9] --->{y_pred[0][0]:.2f}")

