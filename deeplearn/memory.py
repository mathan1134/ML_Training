# # lstm for memory storage 
# # it not store anything just tell pattern to tell which means temper memory to express

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









# lstm predict next word only not continues transformer do that


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer


sentences=[
    "how are you",
    "go to sleep",
    "are you okay",
    "life is good",
    "what you want",
    "risk reward ratio",
    "you are nice"   
]

tokenize=Tokenizer(oov_token="<oov>")
tokenize.fit_on_texts(sentences)

word_index=tokenize.word_index
vacab_size=len(word_index)+1

X=[]
y=[]

for sentence in sentences:
    words=sentence.split()
    for i in range(len(words)-2):
        X.append([word_index[words[i]],word_index[words[i+1]]])
        y.append(word_index[words[i+2]])

X=np.array(X)        
y=np.array(y)

model=Sequential([
    Embedding(input_dim=vacab_size,output_dim=10,input_length=2),
    LSTM(50),
    Dense(vacab_size,activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=["accuracy"])
model.fit(X,y,epochs=500,verbose=0)

def pred_word(word1,word2):
    w1=word_index.get(word1,word_index["<oov>"])
    w2=word_index.get(word2,word_index["<oov>"])

    seq=np.array([[w1,w2]])
    pred=model.predict(seq,verbose=0)
    n_idx=int(np.argmax(pred))
    n_word=tokenize.index_word.get(n_idx,"<UNKNOWN>")
    print(f"{word1},{word2}--->{n_word}")

pred_word("are","you")    




