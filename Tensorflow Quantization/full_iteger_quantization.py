import numpy as np
import tensorflow as tf

saved_model_dir = '/home/kawsar/Desktop/Deep Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/coco_hand_centernet_resnet50/saved_model'

def representative_dataset():
    for _ in range(100):data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()

input_details = tflite_quant_model.get_input_details()
output_details = tflite_quant_model.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

