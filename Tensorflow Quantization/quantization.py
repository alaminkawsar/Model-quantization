import tensorflow as tf


_TFLITE_MODEL_PATH = "/home/kawsar/Desktop/Deep Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/efficientdet_lite2.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model('/home/kawsar/Desktop/Deep Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/efficientdet_lite2_detection')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
signature_details = interpreter.get_signature_list()
print(signature_details)

with open(_TFLITE_MODEL_PATH, 'wb') as f:
  f.write(tflite_model)