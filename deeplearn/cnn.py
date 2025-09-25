import tensorflow as ts
from tensorflow.keras import datasets,layers,models 


(X_train,y_train),(X_test,y_test)=datasets.mnist.load_data()

X_train=X_train.reshape(-1,28,28,1).astype("float")/255.0
X_test=X_test.reshape(-1,28,28,1).astype("float")/255.0

model=models.Sequential([
         layers.Conv2D(16,(3,3),activation="relu",padding="same",input_shape=(28,28,1)),
         layers.MaxPooling2D((2,2)),
         layers.Conv2D(32,(3,3),activation="relu",padding="same"),
         layers.MaxPooling2D(2,2),
         layers.Flatten(),
         layers.Dense(128,activation="relu"),
         layers.Dense(10,activation="softmax")
])


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train,batch_size=64,epochs=3,verbose=1)

test_loss,test_accu=model.evaluate(X_test,y_test)
print(f"test accuracy :{test_accu*100:.2f}%")