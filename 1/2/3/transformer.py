# transform_utils.py
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import base64
from pkg_ebe60a2d_f0be_459a_9cbc_4ef6a9416ad2_fd70695b_4aea_4521_9ce0_09b808836e74 import transformer

response = transformer.post_transform("heelo")


def pre_transform(input):
    print("In Custom pre_transform method")
    print(str(input)[:20], "...", str(input)[-20:])
    img_height = 180
    img_width = 180
    img_array = np.array(Image.open(BytesIO(base64.b64decode(input))).resize((img_height, img_width)))
    transformed_input = img_array.tolist()
    print("Custom pre transformation done")
    print(str(transformed_input)[:20], "...", str(transformed_input)[-20:])
    return transformed_input
#

def post_transform(input):
    print("In Custom post_transform method")
    print(str(input)[:20], "...", str(input)[-20:])
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    scores = tf.nn.softmax(input)
    class_index = np.argmax(scores)
    class_name = class_names[class_index]
    confidence = 100 * scores[class_index]
    confidence_value = float(confidence)
    processed_response = (class_name, confidence_value)
    print("Custom post transformation done")
    print(str(processed_response)[:200], "...", str(processed_response)[-200:])
    print(class_name)
    print(confidence)
    return processed_response

