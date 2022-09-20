from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

#flatbuffer need to be upgraded

_TFLITE_LABEL_PATH = "/home/kawsar/Desktop/Deep_Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/tflite_label_map.txt"

_TFLITE_MODEL_WITH_METADATA_PATH = "/home/kawsar/Desktop/Deep_Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/efficientdetD0_quantized.tflite"
_TFLITE_MODEL_PATH = '/home/kawsar/Desktop/Deep_Learning/Deep-Learning-Model_Quantization/Tensorflow Quantization/efficientdetD0_hand_gesture/efficientdeD0_without_metadata.tflite'

writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(_TFLITE_MODEL_PATH), input_norm_mean=[127.5], 
    input_norm_std=[127.5], label_file_paths=[_TFLITE_LABEL_PATH])
writer_utils.save_file(writer.populate(), _TFLITE_MODEL_WITH_METADATA_PATH)