import streamlit as st
import streamlit.components.v1 as components


####LOTTIE####
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
#####LOTTIE####

#####imagegrid######
import streamlit_imagegrid
from streamlit_imagegrid import requests
from io import BytesIO

######imagegrid#####

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
  st_lottie(lottie_t1, width=199, height=150)

st.sidebar.title("CheXM")
st.sidebar.subheader("Pneumonia Detection")
st.sidebar.markdown("---")
st.sidebar.subheader("#Aplha Version Under Development")
st.sidebar.image("https://img.shields.io/badge/CheXM-v1.0-green.svg", width=100)

components.iframe("https://giphy.com/embed/WfZBqVVywQdd1OloEd", width=300, height=300)

st.sidebar.subheader("About")
st.sidebar.markdown("CheXM is a deep learning model, It is a convolutional neural network (CNN) model that is trained to detect pneumonia in chest X-ray images. This model is trained on 2 Classes CXR X-ray images, Pneumonia and Normal. This tool is based dataset was published by [Paulo Breviglieri](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images), a revised version of [ Paul Mooney's most popular Dataset Published in Cell Journal](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).")
st.sidebar.markdown("---")

st.sidebar.markdown ("Credits: This Model is a fork of [Hardik's X-ray ⚕️ Classifier](https://github.com/smarthardik10/Xray-classifier) ")

st.sidebar.image("https://img.shields.io/badge/Version-Alpha-orange.svg")
#st.sidebar.markdown("CheXM is a deep learning model developed by Pranjal Pruthi. It is a convolutional neural network (CNN) model that is trained to detect pneumonia in chest X-ray images. The model is trained on the Chest X-ray images of the American Heart Association (AHA) and the International Chest Imaging Conference (ICIC) databases. The model is trained on the following modalities: CT, MRI, PET, and X-ray.")


st.sidebar.markdown(
        "[![this is pranjal's kofi link](https://i.imgur.com/XnWftZ9.png)](https://ko-fi.com/pranjalpruthi)"
    )

####lottie logo####


with st.sidebar:
  st_lottie(lottie_ball)


#####LOTTIE####


st.write("""
# X-Ray Classification (Pneumonia/Normal) 

This app counts the disease probability of query CXR(Chest XRay Scan-Frontal)!

***
""")


  
######Uploader########

temp = st.file_uploader("Upload X-Ray Image")

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! that doesn't look like an image. Try again, or dragndrop from examples below ")

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
          
            

####imagegrid####
st.title('Image grid test')
zoom_val = st.sidebar.slider('Zoom',40,240)

urls = [
        {
          "width": 1000,
          "height": 666,
          "src": "https://as2.ftcdn.net/jpg/02/25/53/52/1000_F_225535263_n14yO9cXk18X90qQYxBf5Vcf00uOtAhW.jpg"
        },
        {
          "width": 1000,
          "height": 422,
          "src": "https://as2.ftcdn.net/jpg/02/81/07/83/1000_F_281078312_PcISs3yKL9r70nCqvDkgOR17UBGIw97C.jpg"
        },
        {
          "width": 1000,
          "height": 666,
          "src": "https://as2.ftcdn.net/jpg/02/96/35/64/1000_F_296356423_f6IEidPVRWzaqj2MJQ2VLJJTYGRAtY4P.jpg"
        }
      ]

return_value = streamlit_imagegrid(urls=urls,height=1000)


if return_value is not None:
    response = requests.get(return_value)
    st.sidebar.markdown('<img src={} width=240px></img>'.format(return_value),unsafe_allow_html=True)
#######################  

  
