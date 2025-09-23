# import numpy as np

# def sigmoid(x):
#     return 1/(1+np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1-x)

# X = np.array([[0,0],
#               [0,1],
#               [1,0],
#               [1,1]])

# y = np.array([[0],[1],[1],[0]])

# np.random.seed(42)
# iln=2
# hln=2
# on=1

# w1=np.random.uniform(size=(iln,hln))
# b1=np.random.uniform(size=(1,hln))
# w2=np.random.uniform(size=(hln,on))
# b2=np.random.uniform(size=(1,on))

# epochs=8000
# learning_rate=0.3



# # training
# for _ in range(epochs):
#     hidden_input=np.dot(X,w1)+b1
#     hidden_op=sigmoid(hidden_input)

#     final_input=np.dot(hidden_op,w2)+b2
#     final_op=sigmoid(final_input)

#     # backbrobagation
#     error=y-final_op
#     d_op=error*sigmoid_derivative(final_op)

#     err_hid=d_op.dot(w2.T)
#     d_hidd=err_hid*sigmoid_derivative(hidden_op)


#     # update bias&weights
#     w2 += hidden_op.T.dot(d_op) * learning_rate
#     b2 += np.sum(d_op, axis=0, keepdims=True) * learning_rate
#     w1 += X.T.dot(d_hidd) * learning_rate
#     b1 += np.sum(d_hidd, axis=0, keepdims=True) * learning_rate

# # Final output after training
# print("Predictions after training:")
# print(final_op)



#    words to find


import numpy as np

vocab=["go","now","where","home","stop","please","wait","here"]
word_to_index={word:i for i,word in enumerate(vocab)}
vacab_size=len(vocab)

def s_vector(sentence):
    vec=np.zeros(vacab_size)
    for word in sentence.split():
        if word in word_to_index:
            vec[word_to_index[word]]=1
    return vec

X_sentence=[
    "go now","where home","stop please","wait here"
]
y_labels=np.array([
    [1],[0],[1],[0]
])

X=np.array([s_vector(s) for s in X_sentence])


def sigmoid(x):
    return 1/(1+np.exp(-x))

def dervative_sigmoid(x):
    return x*(1-x)

np.random.seed(1)
ip_n=vacab_size
hn_n=4
op_n=1
l_rate=0.3
epochs=5000

w1=np.random.randn(ip_n,hn_n)
b1=np.zeros((1,hn_n))
w2=np.random.randn(hn_n,op_n)
b2=np.zeros((1,op_n))

for epoch in range(epochs):
    h_in=np.dot(X,w1)+b1
    h_op=sigmoid(h_in)

    f_in=np.dot(h_op,w2)+b2
    f_op=sigmoid(f_in)


    error=y_labels-f_op

    d_op=error*dervative_sigmoid(f_op)
    d_hi=d_op.dot(w2.T)*dervative_sigmoid(h_op)

    w2 +=h_op.T.dot(d_op)* l_rate
    b2 +=np.sum(d_op,axis=0,keepdims=True)*l_rate
    w1 +=X.T.dot(d_hi)* l_rate
    b1 +=np.sum(d_hi,axis=0,keepdims=True)*l_rate

    if epoch % 1000==0:
        print(f"epoch {epoch},loss: {np.mean(np.abs(error)):.4f}")


print("training completed")    
print("\nTraining complete!\n")

for s in X_sentence:
    vec=s_vector(s).reshape(1,-1)
    h_op=sigmoid(np.dot(vec,w1)+b1)
    f_op=sigmoid(np.dot(h_op,w2)+b2)
    print(f"sentences {s}->prediction: {f_op[0][0]:.3f}")
