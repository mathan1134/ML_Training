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

# import tensorflow as tf
# from tensorflow.keras.layers import Input,Dense,MultiHeadAttention,LayerNormalization,GlobalAveragePooling1D,Lambda
# from tensorflow.keras.models import Model
# import numpy as np

# def t_encoder(x,num_heads,ff_dim):
#     attn=MultiHeadAttention(num_heads=num_heads,key_dim=ff_dim)(x,x)
#     x=LayerNormalization()(x+attn)

#     ff=Dense(ff_dim,activation="relu")(x)
#     ff=Dense(x.shape[-1])(ff)
#     return LayerNormalization()(x+ff)

# input=Input(shape=(5,16))
# x=t_encoder(input,num_heads=2,ff_dim=32)
# # x = GlobalAveragePooling1D()(x)
# x = Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
# op=Dense(2,activation="softmax")(x)

# bert_model=Model(input,op)
# bert_model.summary()

# X=np.random.rand(10,5,16)
# y=tf.keras.utils.to_categorical(np.random.randint(0,2,10),num_classes=2)

# bert_model.compile(optimizer="adam",loss="categorical_crossentropy")
# bert_model.fit(X,y,epochs=5)





# gpt (response)

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.layers import Input,Dense,MultiHeadAttention,LayerNormalization,Dropout
# from tensorflow.keras.models import Model

# def t_decoder(x,num_heads,ff_dim):
#     attn=MultiHeadAttention(num_heads=num_heads,key_dim=ff_dim)(x,x,use_causal_mask=True)
#     x=LayerNormalization()(x+attn)

#     ff=Dense(ff_dim,activation="relu")(x)
#     ff=Dense(x.shape[-1])(ff)
#     return LayerNormalization()(x+ff)

# inputs=Input(shape=(5,16))
# x=t_decoder(inputs,num_heads=2,ff_dim=32)
# outputs=Dense(16,activation="linear")(x)

# gpt_model=Model(inputs,outputs)
# gpt_model.summary()

# X=np.random.rand(10,5,16)
# y=np.random.rand(10,5,16)

# gpt_model.compile(optimizer="adam",loss="mse")
# gpt_model.fit(X,y,epochs=5)


