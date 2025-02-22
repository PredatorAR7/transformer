import numpy as np
from PIL import Image
from io import BytesIO
import base64

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def pre_process(input):
    img_height = 180
    img_width = 180
    # Decode the input, load as an image, resize, and convert to a NumPy array
    img_array = np.array(Image.open(BytesIO(base64.b64decode(input))).resize((img_height, img_width)))
    # Ensure the array has a batch dimension
    transformed_input = np.expand_dims(img_array, 0)
    return transformed_input

def post_process(input_scores):
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # Apply softmax to the input scores
    probabilities = softmax(np.array(input_scores))
    # Find the index of the maximum score/probability
    class_index = np.argmax(probabilities)
    class_name = class_names[class_index]
    confidence = 100 * probabilities[class_index]
    # Convert confidence to float for consistency
    confidence_value = float(confidence)
    processed_response = (class_name, confidence_value)
    return processed_response
