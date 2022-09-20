from pickletools import uint8
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from PIL import Image

dictionary = {5: 5, 4: 4, 3: 2, 2: 2, 1: 1}
for key, label in dictionary.as_dict().items():
    print(key,label)