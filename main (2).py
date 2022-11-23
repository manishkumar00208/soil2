
import numpy as np
import tensorflow as tf
import cv2
import pathlib

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:/Users/Manish/Downloads/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
from tensorflow.keras.preprocessing import image
# Test the model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
path = 'C:/Users/Manish/Downloads/alluvial/download (4).jpg'
img = image.load_img(path, target_size=(256, 256))
x = image.img_to_array(img)
input_data = np.expand_dims(x, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()




# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
classes = {0:'Black Soil', 1:'Cinder Soil', 2:'Laterite Soil', 3:'Peat Soil', 4:'Yellow Soil'}
MaxPosition=np.argmax(output_data)
print(classes[MaxPosition])


if(classes[MaxPosition] == "Black Soil"):
    print("Average Soil Organic Matter is ")