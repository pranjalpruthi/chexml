# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from email.mime import image
from PIL import Image
import torch
from torchvision import models, transforms
import streamlit as st
import torchxrayvision as xrv

from tensorflow.keras.utils import img_to_array



st.set_page_config(
        page_title="ResNet50 Multi Class Classfier",
        page_icon="📊",layout="wide"
    )


# set title of app
st.title("MultiClass Classification Application")
st.write("")


col1, col2 = st.columns(2)
with col1:
    st.header("Please provide image to analyze")


# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["jpg", "png","jpeg"])




with col2:
    
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
    resnet =xrv.models.ResNet(weights="resnet50-res512-all")
    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            ),
        transforms.Grayscale(num_output_channels=1)
        ])



    # load the image, pre-process it, and make predictions

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)
    pred = torch.argmax(out, 1)
    return pred



