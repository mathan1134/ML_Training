# bert understand
# gpt response

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense,Input,Embedding,Dropout,LayerNormalization,MultiHeadAttention

# v_size=50
# max_lnth=5
# embd_dim=16
# num_head=2
# ff_dim=32

# inputs=Input(shape=(max_lnth,))

# x=Embedding(input_dim=v_size,output_dim=embd_dim)(inputs)

# atn_op=MultiHeadAttention(num_heads=num_head,key_dim=embd_dim)(x,x)

# x=LayerNormalization()(x+atn_op)

# ff=Dense(ff_dim,activation="relu")(x)
# x=LayerNormalization()(x+ff)

# op=Dense(v_size,activation="softmax")(x)

# model=Model(inputs=inputs,output=op)
# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")
# model.summary()


# X=np.random.randint(0,v_size,size=(100,max_lnth))
# y=np.roll(X,shift=-1,axis=1)

# model.fit(X,y,epochs=500,verbose=1)


# s_ip=np.array([[1,5,3,2,4]])
# pred=model.predict(s_ip)
# n_word=np.argmax(pred[0,-1])
# print("predict next word",(n_word))




# bert for input dtls

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,MultiHeadAttention,LayerNormalization,Dropout
from tensorflow.keras.models import Model
import numpy as np

def t_encoder(x,n_head,ff_dim):
    attn=MultiHeadAttention(n_head=n_head,key_dim=ff_dim)(x,x)
    x=LayerNormalization()(x+attn)

    ff=Dense(ff_dim,activation="relu")(x)
    ff=Dense(x.shape[-1])(ff)
    return LayerNormalization((x+ff))

input=Input(shape=(5,16))
x=t_encoder(input,n_head=2,ff_dim=32)
x=tf.reduce_mean(x,axis=1)
op=Dense(2,activation="softmax")(x)

bert_model=Model(input,op)
bert_model.summary()

X=np.random.rand(10,5,15)
y=tf.keras.utils.to_categorical(np.random.randint(0,2,10),num_clsses=2)

bert_model.compile(optimizer="adam",loss="categorical_crossentrophy",num_classes=2)
bert_model.fit(X,y,epochs=5)


# gpt (response)

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input,Dense,MultiHeadAttention,LayerNormalization,Dropout
from tensorflow.keras.models import Model