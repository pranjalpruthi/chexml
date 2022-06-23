from codecs import ascii_encode
import streamlit as st

# import libraries
from PIL import Image
import torch
from torchvision import models, transforms
import streamlit as st
import torchxrayvision as xrv
import requests
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import numpy as np
import plotly.figure_factory as ff
###
#import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import time
from streamlit_lottie import st_lottie_spinner

###





components.iframe("https://giphy.com/embed/WfZBqVVywQdd1OloEd", width=300, height=300)




# set title of app
st.title("MultiClass Classification Application")
st.write("")
st.markdown("---")


##lottie

####lottie functions####

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_ball ="https://assets4.lottiefiles.com/packages/lf20_dp8xyjzi.json"
lottie_url_t1 ="https://assets10.lottiefiles.com/packages/lf20_nm1huacl.json"
lottie_progress = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_h4th9ofg.json")
lottie_success = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_0unee7cs.json")

lottie_ball = load_lottieurl(lottie_url_ball)
lottie_t1 = load_lottieurl(lottie_url_t1)

##lottie





col1, col2 = st.columns(2)
with col1:
    st.header("Please provide an image to anaylze")


# enable users to upload images for the model to make predictions
    col1.xray = st.file_uploader("Upload an 🩻👇 image", type = ["jpg", "png","jpeg"])




with col2:
    st.header("Predictions")
with col2.empty():
    with st_lottie_spinner(lottie_progress,width=101,height=101, loop=True, key="progress"):
        time.sleep(10)
    st_lottie(lottie_success,width=101,height=101, loop=True, key="success")



def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    model =  xrv.models.ResNet(weights="resnet50-res512-all")







    # transform the input image through resizing, normalization
    transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
        ),
    transforms.Grayscale(num_output_channels=1)
    ])



    # load the image, pre-process it, and make predictions

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    out = model(batch_t)


    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0]*1000
    _, indices = torch.sort(out, descending = True)
    return [(xrv.datasets.default_pathologies[idx], prob[idx].item()) for idx in indices[0][:]]


if col1.xray is not None:
    # display image that user uploaded
    image = Image.open(col1.xray)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)


    
    st.write("")
    st.markdown("The image was successfully uploaded.")
    st.write("💁🏻‍♂️ Just a second ...🤖 Model is predicting...")
    labels = predict(col1.xray)

with col2:

    # print out the top 5 prediction labels with scores
        st.info("🧐 Focus on prediction labels with score above 50%... for relevance")
        st.warning(
            """
            ⚠️Caution: Avoid Inserting Large Size or Unrelevent Images 👨‍⚕️.
    """
        )

        for i in labels:
            st.write("Prediction", i[0], ",  Confidence Score: ", i[1])



###SIdebar


with st.sidebar:
    st_lottie(lottie_t1, width=199, height=150)


st.sidebar.title("CheXM Multi Class Prediction on CXR Scans")
st.sidebar.subheader("MultiClass Prediction ")
st.sidebar.markdown("---")
st.sidebar.subheader("#Alpha Version Under Development")
st.sidebar.image("https://img.shields.io/badge/CheXM-v1.0-green.svg", width=100)
st.sidebar.image("https://i.imgur.com/PEbwL1c.png", width=200)

st.sidebar.subheader("About")
st.sidebar.info("CheXM is a deep learning model, It is a convolutional neural network (ResNet50) model that is trained to detect pneumonia in chest X-ray images. This model is trained on 18 Classes CXR X-ray images,  This tool is based on various training datasets  was published by [Paulo Breviglieri](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images), a revised version of [ Paul Mooney's most popular Dataset Published in Cell Journal](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).")
st.sidebar.markdown("---")

st.sidebar.markdown ("Credits: This Model is a Modified fork of [Hardik's X-ray ⚕️ Classifier](https://github.com/smarthardik10/Xray-classifier) ")

st.sidebar.image("https://img.shields.io/badge/Version-Alpha-orange.svg")
#st.sidebar.markdown("CheXM is a deep learning model developed by Pranjal Pruthi. It is a convolutional neural network (CNN) model that is trained to detect pneumonia in chest X-ray images. The model is trained on the Chest X-ray images of the American Heart Association (AHA) and the International Chest Imaging Conference (ICIC) databases. The model is trained on the following modalities: CT, MRI, PET, and X-ray.")

st.sidebar.info(
        """
        This app is maintained by Pranjal Pruthi. You can learn more about me at
        [linkedin.com/in/pranjal-pruthi](https://www.linkedin.com/in/pranjal-pruthi).
"""
    )
st.sidebar.markdown(
        "[![this is pranjal's kofi link](https://i.imgur.com/XnWftZ9.png)](https://ko-fi.com/pranjalpruthi)"
    )
with st.sidebar:
    st_lottie(lottie_ball)

####lottie logo####
    

#####LOTTIE####
st.info(""" ###So, what do these categories mean? This is actually explained in the paper linked on the [Know More](https://share.streamlit.io/pranjalpruthi/chest-xray-analysis-ml/main/%F0%9F%93%8A_CheXM_Multi_Class_Prediction.py/Know_More!!!) Section. **👈."):

- Aortic enlargement: "An abnormal bulge that occurs in the wall of the major blood vessel."

- Atelectasis: "Collapse of a part of the lung due to a decrease in the amount of air in the alveoli resulting in volume loss and increased density." (see also Merck Manual)

- Calcification: "Deposition of calcium salts in the lung." - one article I looked at says that "[...] calcifications occur in a damaged lung following an inflammatory process such as infection (tuberculosis, histoplasmosis, Pneumocystis carnii), bleeding or pulmonary infarction" (Bendayan et al. 2000)

- Cardiomegaly: "Enlargement of the heart, occurs when the heart of an adult patient is larger than normal and the cardiothoracic ratio is greater than 0.5."

- Consolidation: "Any pathologic process that fills the alveoli with fluid, pus, blood, cells (including tumor cells) or other substances resulting in lobar, diffuse or multifocal ill-defined opacities."

- ILD: "Interstitial lung disease (ILD) Involvement of the supporting tissue of the lung parenchyma resulting in fine or coarse reticular opacities or small nodules." (see also Merck Manual)

- Infiltration: "An abnormal substance that accumulates gradually within cells or body tissues or any substance or type of cell that occurs within or spreads as through the interstices (interstitium and/or alveoli) of the lung, that is foreign to the lung, or that accumulates in greater than normal quantity within it."

- Lung Opacity: "Any abnormal focal or generalized opacity or opacities in lung fields (blanket tag including but not limited to consolidation, cavity, fibrosis, nodule, mass, calcification, interstitial thickening, etc.)."

- Nodule/Mass: "Any space occupying lesion either solitary or multiple."

- Other lesion: "Other lesions that are not on the list of findings or abnormalities mentioned above."

- Pleural effusion: "Abnormal accumulations of fluid within the pleural space." (see also Merck Manual)

- Pleural thickening: "Any form of thickening involving either the parietal or visceral pleura."

- Pneumothorax: "The presence of gas (air) in the pleural space." (see also the Merck Manual)

- Pulmonary fibrosis: "An excess of fibrotic tissue in the lung."
""")



