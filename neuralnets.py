from keras import Sequential, layers
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.initializers import initializers_v2
import tensorflow as tf


data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

model = Sequential(
    [layers.Dense(32, activation='relu', input_shape=[X.shape[1]], kernel_initializer=initializers_v2.GlorotNormal(), use_bias=True),
     layers.Dense(units=1, activation='sigmoid')])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=100, epochs=100)

predictions = model.predict(X_test, batch_size=10, use_multiprocessing=True)

prediction = np.array([1 if x > 0.5 else 0 for x in predictions])

print(np.sum(prediction == y_test)/len(y_test))

history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
plt.show()
