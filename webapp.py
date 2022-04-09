import streamlit as st
import streamlit.components.v1 as components

####LOTTIE####
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
#####LOTTIE####

import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "cnn_pneu_vamp_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()

####lottie functions####

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_ball ="https://assets4.lottiefiles.com/packages/lf20_dp8xyjzi.json"
lottie_url_t1 ="https://assets10.lottiefiles.com/packages/lf20_nm1huacl.json"

lottie_ball = load_lottieurl(lottie_url_ball)
lottie_t1 = load_lottieurl(lottie_url_t1)

######################
# Page Title
######################

##lottie###
with st.sidebar:
  st_lottie(lottie_t1)

st.sidebar.title("CheXM")
st.sidebar.subheader("Pneumonia Detection")
st.sidebar.markdown("---")
st.sidebar.subheader("#Beta Version Under Development")
st.sidebar.image("https://img.shields.io/badge/CheXM-v1.0-green.svg", width=100)

components.iframe("https://giphy.com/embed/WfZBqVVywQdd1OloEd", width=200, height=200)

st.sidebar.subheader("About")
st.sidebar.markdown("CheXM is a deep learning model  , It is a convolutional neural network (CNN) model that is trained to detect pneumonia in chest X-ray images. It consists of 2 categories, Pneumonia and Normal. This dataset was published by Paulo Breviglieri, a revised version of Paul Mooney's most popular dataset. databases. The model is trained 2 Classes CXR X-ray images.")
st.sidebar.markdown("---")
st.sidebar.markdown ("Fork of <a href="https://github.com/smarthardik10/Xray-classifier">Credit:Hardik's X-ray ⚕️ Classifier </a>")
st.sidebar.image("https://img.shields.io/badge/Version-Beta-orange.svg")
#st.sidebar.markdown("CheX2 is a deep learning model developed by Pranjal Pruthi. It is a convolutional neural network (CNN) model that is trained to detect pneumonia in chest X-ray images. The model is trained on the Chest X-ray images of the American Heart Association (AHA) and the International Chest Imaging Conference (ICIC) databases. The model is trained on the following modalities: CT, MRI, PET, and X-ray.")

####lottie logo####


with st.sidebar:
  st_lottie(lottie_ball)


#####LOTTIE####


st.write("""
# X-Ray Classification (Pneumonia/Normal) 

This app counts the disease probability of query CXR(Chest XRay Scan-Frontal)!

***
""")


  


temp = st.file_uploader("Upload X-Ray Image")

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! that doesn't look like an image. Try again.")

else:

 

  hardik_img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  pp_hardik_img = image.img_to_array(hardik_img)
  pp_hardik_img = pp_hardik_img/255
  pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

  #predict
  hardik_preds= cnn.predict(pp_hardik_img)
  if hardik_preds>= 0.5:
    out = ('I am {:.2%} percent confirmed that this is a Pneumonia case'.format(hardik_preds[0][0]))
  
  else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-hardik_preds[0][0]))

  st.success(out)
  
  image = Image.open(temp)
  st.image(image,use_column_width=True)
          
            

  

  
