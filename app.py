import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.models.load_model('model_v1.h5')
  return model

model = load_model()

def show_predict_page():
  st.write('hey')
  
  file = st.file_uploader("Upload Brain MRI", type=['jpg','png','jpeg'])
  
  def pedict(model, img):
    size = (256,256)
    image = ImageOps.fit(img, size, Image.ANTIALIAS)
    img_array = np.array(image)
    img_reshape = img_array[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

  image = Image.open(file)
  st.image(image, use_column_width=True)
  predictions = pedict(model,image)
  class_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
  string = class_names[np.argmax(predictions)]
  st.success(string)

show_predict_page()