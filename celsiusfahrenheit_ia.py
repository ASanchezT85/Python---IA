import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# layer = tf.keras.layers.Dense(units=1, input_shape=[1])
# model = tf.keras.Sequential([layer])

hiddenLayer1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hiddenLayer2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hiddenLayer1, hiddenLayer2, output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
)

print('Comenzar entrenamienot...')
record = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print('Modelo entrenado!')

plt.xlabel('# Epoca')
plt.ylabel('# Magnitud de perdida')
plt.plot(record.history['loss'])

print('Hagamos una prediccion')
resul = model.predict([100.0])
print('El resultado es ' + str(resul) + ' fahrenheit')

# print('Variable internas del modelo')
# print(layer.get_weights())
