import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],[1],[1],[0]])

np.random.seed(42)
iln=2
hln=2
on=1

w1=np.random.uniform(size=(iln,hln))
b1=np.random.uniform(size=(1,hln))
w2=np.random.uniform(size=(hln,on))
b2=np.random.uniform(size=(1,on))

epochs=8000
learning_rate=0.3



# training
for _ in range(epochs):
    hidden_input=np.dot(X,w1)+b1
    hidden_op=sigmoid(hidden_input)

    final_input=np.dot(hidden_op,w2)+b2
    final_op=sigmoid(final_input)

    # backbrobagation
    error=y-final_op
    d_op=error*sigmoid_derivative(final_op)

    err_hid=d_op.dot(w2.T)
    d_hidd=err_hid*sigmoid_derivative(hidden_op)


    # update bias&weights
    w2 += hidden_op.T.dot(d_op) * learning_rate
    b2 += np.sum(d_op, axis=0, keepdims=True) * learning_rate
    w1 += X.T.dot(d_hidd) * learning_rate
    b1 += np.sum(d_hidd, axis=0, keepdims=True) * learning_rate

# Final output after training
print("Predictions after training:")
print(final_op)