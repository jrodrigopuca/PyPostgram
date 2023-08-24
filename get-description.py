import tensorflow as tf
import numpy as np
tf.compat.v1.enable_eager_execution()
# Cargar el modelo pre-entrenado InceptionV3
model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)

img = tf.image.decode_image(tf.io.read_file("agua.jpg"))
img = tf.image.resize(img, (299, 299))
img = tf.cast(img, dtype=tf.float32)
img = tf.image.per_image_standardization(img)

img = tf.keras.applications.inception_v3.preprocess_input(img)

# Realizar la predicción usando el modelo
predictions = model.predict(img)

# Obtenemos la clase con la mayor probabilidad
class_id = np.argmax(predictions)

# Imprimimos la clase de la imagen
print(tf.keras.applications.inception_v3.decode_predictions(predictions)[0][1])
