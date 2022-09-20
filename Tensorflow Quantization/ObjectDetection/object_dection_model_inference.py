from pickletools import uint8
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#image path
img_path = '/home/kawsar/Desktop/Deep Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/ObjectDetection/object_img.jpg'
img = plt.imread(img_path)
print("image shape=",img.shape)
img = cv2.resize(img,(300,300))
print("image shape=",img.shape)

#model path
model_path = '/home/kawsar/Desktop/Deep Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/coco_hand_centernet_resnet50/hand_detector_centernet_resnet50_quant.tflite'
# Load quantized TFLite model
tflite_interpreter_quant = tf.lite.Interpreter(model_path=model_path)

# Learn about its input and output details
input_details = tflite_interpreter_quant.get_input_details()
output_details = tflite_interpreter_quant.get_output_details()

# tflite_interpreter_quant.resize_tensor_input(output_details[0]['index'], (32, 5))
# # Resize input and output tensors to handle image size
tflite_interpreter_quant.resize_tensor_input(input_details[0]['index'], (1, 300, 300, 3))
tflite_interpreter_quant.allocate_tensors()

input_details = tflite_interpreter_quant.get_input_details()
output_details = tflite_interpreter_quant.get_output_details()

# print("== Input details ==")
# print("name:", input_details[0]['name'])
# print("shape:", input_details[0]['shape'])
# print("type:", input_details[0]['dtype'])

# print("\n== Output details ==")
# print("name:", output_details[0]['name'])
# print("shape:", output_details[0]['shape'])
# print("type:", output_details[0]['dtype'])

# Test model on random input data.
# input_shape = input_details[0]['shape']
# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()

# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
print(img.dtype)
img = np.expand_dims(img, axis=0)

tflite_interpreter_quant.set_tensor(input_details[0]['index'], img)
print("set tensor done")
tflite_interpreter_quant.invoke()
print("invoke done")
tflite_q_model_predictions = tflite_interpreter_quant.get_tensor(output_details[0]['index'])
print("\nPrediction results shape:", tflite_q_model_predictions.shape)
print(tflite_q_model_predictions)
# plt.imshow(img)
# plt.show()

