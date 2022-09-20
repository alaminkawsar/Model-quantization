# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
model_path = '/home/kawsar/Desktop/Deep Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/coco_hand_ssd_mobilenet_v2_fpn_320/hand_detector_quant.tflite'
base_options = core.BaseOptions(file_name=model_path)
detection_options = processor.DetectionOptions(max_results=2)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
# detector = vision.ObjectDetector.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)