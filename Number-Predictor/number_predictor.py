import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Activation
from keras.utils import plot_model
import matplotlib.pyplot as plt

# Load mnist dataset from Keras
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = Sequential()
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10,activation=tf.nn.softmax))
# 10 for 10 numbers i.e 0 - 9
# Compile the model
model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=10)

val_loss , val_acc = model.evaluate(x_test,y_test)
print("Validation Loss : "+str(val_loss))
print("Validation Accuracy : "+str(val_acc))

plt.imshow(x_train[0],cmap = plt.cm.binary)
plt.show()
# print(x_train[0])
print(y_train[0])

model.save("number_read.h5")

new_model = tf.keras.models.load_model("number_read.h5")

predictions = new_model.predict(x_test)

print(np.argmax(predictions[1]))

plt.imshow(x_test[1],cmap = plt.cm.binary)
plt.show()

plot_model(model, to_file='num_reader_model.png')
