import tensorflow as tf

saved_model_dir = '/home/kawsar/Desktop/Deep Learning/Tensorflow Quantization/coco_hand_centernet_resnet50/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_quant_model = converter.convert()

with open("./coco_hand_centernet_resnet50/hand_detector_centernet_resnet50_quant.tflite","wb") as f:
    f.write(tflite_quant_model)
