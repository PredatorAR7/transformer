import numpy as np
from PIL import Image
from io import BytesIO
import base64


def softmax(x):
   """Compute softmax values for each set of scores in x."""
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum(axis=0)


def pre_transform(input):
   print("In Custom pre_transform method")
   # Displaying initial part of the input for debugging
   print(str(input)[:20], "...", str(input)[-20:])
   img_height = 180
   img_width = 180
   # Decode the input, load as an image, resize, and convert to a NumPy array
   img_array = np.array(Image.open(BytesIO(base64.b64decode(input))).resize((img_height, img_width)))
   transformed_input = img_array.tolist()
   print("Custom pre transformation done")
   # Displaying initial part of the transformed input for debugging
   print(str(transformed_input)[:20], "...", str(transformed_input)[-20:])
   return transformed_input



