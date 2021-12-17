
import tensorflow as tf
mnist=tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train,x_test=x_train/255.,x_test/255.

model=tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28,28)),
     tf.keras.layers.Dense(512,activation=tf.nn.relu),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10,activation=tf.nn.softmax)
                                  ])

# we can do this instead
'''
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential
model1=Sequential([
     Flatten(input_shape=(28,28)),
     Dense(512,activation=tf.nn.relu),
     Dropout(0.2),
     Dense(10,activation=tf.nn.softmax)
                                  ])
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

