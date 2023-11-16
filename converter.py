import tensorflow as tf

model = tf.keras.models.load_model('models/my_mobilenetv2.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("models/my_mobilenetv2.tflite","wb").write(tflite_model)

