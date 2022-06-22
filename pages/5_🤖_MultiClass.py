from codecs import ascii_encode
import streamlit as st
# import libraries
from email.mime import image
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

import skimage
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms


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







# enable users to upload images for the model to make predictions

xray = st.file_uploader("Upload an 🩻👇 image", type = ["jpg", "png","jpeg"])



    # create a ResNet model
model =  xrv.models.ResNet(weights="resnet50-res512-all")



img = skimage.io.imread(xray)
img = xrv.datasets.normalize(img, 255)

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

# Add color channel
img = img[None, :, :]

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

img = transform(img)

with torch.no_grad():
    img = torch.from_numpy(img).unsqueeze(0)
    preds = model(img).cpu()
    output = {
        k: float(v*100)
        for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
    }
st.write(output)










    # return the top 5 predictions ranked by highest probabilities

# display image that user uploaded
img = Image.open(xray)
st.image(img, caption = 'Uploaded Image.', use_column_width = True)
st.write("")
st.markdown("The image was successfully uploaded.")




###Sidebar


